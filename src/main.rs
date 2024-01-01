const MEMORY_SIZE: usize = 4096;
const WIDTH: usize = 64;
const HEIGHT: usize = 32;
const VIEWPORT_MULTIPLYER: f64 = 25.0;
const FRAME_RATE: f64 = 60.0;

use std::env;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

use std::{
    sync::mpsc::{self, Receiver, Sender},
    thread,
};

use pixels::{Pixels, SurfaceTexture};
use winit::{
    dpi::LogicalSize,
    event::{Event, VirtualKeyCode},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use winit_input_helper::WinitInputHelper;

const FONT: [[u8; 5]; 16] = [
    [0xF0, 0x90, 0x90, 0x90, 0xF0], // 0
    [0x20, 0x60, 0x20, 0x20, 0x70], // 1
    [0xF0, 0x10, 0xF0, 0x80, 0xF0], // 2
    [0xF0, 0x10, 0xF0, 0x10, 0xF0], // 3
    [0x90, 0x90, 0xF0, 0x10, 0x10], // 4
    [0xF0, 0x80, 0xF0, 0x10, 0xF0], // 5
    [0xF0, 0x80, 0xF0, 0x90, 0xF0], // 6
    [0xF0, 0x10, 0x20, 0x40, 0x40], // 7
    [0xF0, 0x90, 0xF0, 0x90, 0xF0], // 8
    [0xF0, 0x90, 0xF0, 0x10, 0xF0], // 9
    [0xF0, 0x90, 0xF0, 0x90, 0x90], // A
    [0xE0, 0x90, 0xE0, 0x90, 0xE0], // B
    [0xF0, 0x80, 0x80, 0x80, 0xF0], // C
    [0xE0, 0x90, 0x90, 0x90, 0xE0], // D
    [0xF0, 0x80, 0xF0, 0x80, 0xF0], // E
    [0xF0, 0x80, 0xF0, 0x80, 0x80], // F
];

static CHIP_INPUT: [AtomicBool; 16] = [
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
];

const ON_COLOR_RGBA: [u8; 4] = [251, 205, 56, 0xFF];
const OFF_COLOR_RGBA: [u8; 4] = [148, 104, 29, 0xFF];

struct Chip8 {
    sender: Sender<[bool; WIDTH * HEIGHT]>, // Non-specified data;
    pixel_buffer: [bool; WIDTH * HEIGHT],
    debug_string: String,

    memory: [u8; MEMORY_SIZE],
    program_counter: u16,
    register_i: u16,
    delay_timer: u8,
    sound_timer: u8,
    registers: [u8; 16],
    stack: [u16; 16],
    stack_pointer: u16,
}
impl Chip8 {
    fn new(rom: Vec<u8>, sender: Sender<[bool; WIDTH * HEIGHT]>) -> Self {
        let mut memory = [0u8; MEMORY_SIZE];
        for (i, byte) in rom.iter().enumerate() {
            memory[i + 512] = *byte;
        }
        let mut i = 0;
        for font in FONT {
            for byte in font {
                memory[i] = byte;
                i += 1;
            }
        }
        let chip8 = Chip8 {
            sender,
            pixel_buffer: [false; WIDTH * HEIGHT],
            debug_string: "".to_string(),
            memory,
            program_counter: 512,
            register_i: 0,
            delay_timer: 0,
            sound_timer: 0,
            registers: [0; 16],
            stack: [0; 16],
            stack_pointer: 0,
        };
        chip8.redraw();
        return chip8;
    }
    fn set_carry(&mut self, carry: bool) {
        self.registers[0x0F] = carry as u8
    }
    fn redraw(&self) {
        self.sender.send(self.pixel_buffer).unwrap();
    }
    fn arithmetic(&mut self, opcode: usize, x: usize, y: usize) {
        match opcode {
            0x0 => {
                self.registers[x] = self.registers[y];
                self.set_carry(false);
            }
            0x1 => {
                self.registers[x] |= self.registers[y];
                self.set_carry(false);
            }
            0x2 => {
                self.registers[x] &= self.registers[y];
                self.set_carry(false);
            }
            0x3 => {
                self.registers[x] ^= self.registers[y];
                self.set_carry(false);
            }
            0x4 => {
                let overflow = (self.registers[x] as usize + self.registers[y] as usize) > 255;
                self.registers[x] = self.registers[x].wrapping_add(self.registers[y]);
                self.set_carry(overflow);
            }
            0x5 => {
                let overflow = self.registers[x] >= self.registers[y];
                self.registers[x] = self.registers[x].wrapping_sub(self.registers[y]);
                self.set_carry(overflow);
            }
            0x6 => {
                let overflow = (self.registers[x] & 0b0000_0001) != 0;
                self.registers[x] = self.registers[y] >> 1;
                self.set_carry(overflow);
            }
            0x7 => {
                let overflow = self.registers[x] <= self.registers[y];
                self.registers[x] = self.registers[y].wrapping_sub(self.registers[x]);
                self.set_carry(overflow);
            }
            0xE => {
                let overflow = (self.registers[x] & 0b1000_0000) != 0;
                self.registers[x] = self.registers[y] << 1;
                self.set_carry(overflow);
            }
            _ => panic!("{}", self.debug_string),
        }
    }
    fn run(&mut self) {
        const FRAME_TIME: u128 = (1_000_000_000.0 / FRAME_RATE) as u128;
        let mut accumulator = 0;
        let mut last_time = Instant::now();
        loop {
            // TIMING
            let current_time = Instant::now();
            let delta = current_time - last_time;
            accumulator += delta.as_nanos();
            last_time = current_time;
            let frames_elapsed = accumulator / FRAME_TIME;
            accumulator -= frames_elapsed * FRAME_TIME;

            if self.delay_timer > 0 {
                self.delay_timer =
                    std::cmp::max(0, self.delay_timer as i32 - frames_elapsed as i32) as u8;
            }
            // TIMING

            let instruction: [u8; 2] = [
                self.memory[self.program_counter as usize],
                self.memory[self.program_counter as usize + 1],
            ];

            // Converting to usize now avoids type juggling.
            let i: usize = ((instruction[0] & 0xF0) >> 4) as usize; // First nibble, 4 bits
            let x: usize = (instruction[0] & 0x0F) as usize; // Second nibble, 4 bits
            let y: usize = ((instruction[1] & 0xF0) >> 4) as usize; // Third nibble, 4 bits
            let n: usize = (instruction[1] & 0x0F) as usize; // Fourth nibble, 4 bits
            let nn: u8 = instruction[1]; // Second byte, 8 bits
            let nnn: u16 = u16::from(x as u8) << 8 | u16::from(nn); // third, fourth, fifth nibble, 12 bits.

            self.debug_string = format!(
                "Interpreter failed at instruction: {:X}{:X}{:X}{:X}",
                i, x, y, n
            );

            match (i, x, y, n) {
                (0x0, 0x0, 0xE, 0x0) => {
                    self.pixel_buffer = [false; WIDTH * HEIGHT];
                    self.redraw();
                }
                (0x0, 0x0, 0xE, 0xE) => {
                    self.program_counter = self.stack[self.stack_pointer as usize];
                    self.stack_pointer -= 1;
                }
                (0x1, _, _, _) => {
                    self.program_counter = nnn;
                    continue;
                }
                (0x2, _, _, _) => {
                    self.stack_pointer += 1;
                    self.stack[self.stack_pointer as usize] = self.program_counter;
                    self.program_counter = nnn;
                    continue;
                }
                (0x8, _, _, _) => self.arithmetic(n, x, y),
                (0x3, _, _, _) => {
                    if self.registers[x] == nn {
                        self.program_counter += 2;
                    }
                }
                (0x4, _, _, _) => {
                    if self.registers[x] != nn {
                        self.program_counter += 2;
                    }
                }
                (0x5, _, _, _) => {
                    if self.registers[x] == self.registers[y] {
                        self.program_counter += 2;
                    }
                }
                (0x6, _, _, _) => {
                    self.registers[x] = nn;
                }
                (0x7, _, _, _) => self.registers[x] = self.registers[x].wrapping_add(nn),
                (0x9, _, _, _) => {
                    if self.registers[x] != self.registers[y] {
                        self.program_counter += 2;
                    }
                }
                (0xA, _, _, _) => {
                    self.register_i = nnn;
                }
                (0xB, _, _, _) => {
                    self.program_counter = nnn + self.registers[0] as u16;
                    continue;
                }
                (0xC, _, _, _) => self.registers[x] = fastrand::u8(..) & nn,
                (0xD, _, _, _) => {
                    let rx = self.registers[x] as usize % WIDTH;
                    let ry = self.registers[y] as usize % HEIGHT;

                    self.set_carry(false);

                    for row in 0..n {
                        let y = ry + row;
                        if y >= HEIGHT {
                            break;
                        }

                        let byte = self.memory[self.register_i as usize + row];
                        for col in 0..8 {
                            let x = rx + col;
                            if x >= WIDTH {
                                break;
                            }

                            let old_pixel = self.pixel_buffer[x + y * WIDTH];
                            let new_pixel = (byte & (1 << (7 - col))) != 0;
                            self.pixel_buffer[x + y * WIDTH] ^= new_pixel;

                            if old_pixel && new_pixel {
                                self.set_carry(true);
                            }
                        }
                    }
                    self.redraw();
                }
                (0xE, _, 0x9, 0xE) => {
                    if CHIP_INPUT[self.registers[x] as usize].load(Relaxed) {
                        self.program_counter += 2;
                    }
                }
                (0xE, _, 0xA, 0x1) => {
                    if !CHIP_INPUT[self.registers[x] as usize].load(Relaxed) {
                        self.program_counter += 2;
                    }
                }
                (0xF, _, 0x5, 0x5) => {
                    for i in 0..x + 1 {
                        self.memory[self.register_i as usize + i] = self.registers[i];
                    }
                    self.register_i += 1;
                }
                (0xF, _, 0x6, 0x5) => {
                    for i in 0..x + 1 {
                        self.registers[i] = self.memory[self.register_i as usize + i];
                    }
                    self.register_i += 1;
                }
                (0xF, _, 0x3, 0x3) => {
                    let mut num = self.registers[x];
                    let num_ones = num % 10;
                    num -= num_ones;
                    let num_tens = (num % 100) / 10;
                    num -= num_tens;
                    let num_hundreds = num / 100;

                    self.memory[self.register_i as usize] = num_hundreds;
                    self.memory[self.register_i as usize + 1] = num_tens;
                    self.memory[self.register_i as usize + 2] = num_ones;
                }
                (0xF, _, 0x1, 0xe) => self.register_i += self.registers[x] as u16,
                (0xF, _, 0x2, 0x9) => self.register_i = u16::from(self.registers[x] * 5),
                (0xF, _, 0x0, 0x7) => self.registers[x] = self.delay_timer,
                (0xF, _, 0x1, 0x5) => self.delay_timer = self.registers[x],
                (0xF, _, 0x1, 0x8) => self.sound_timer = self.registers[x],
                (0xF, _, 0x0, 0xA) => {} // Doesn't halt, should.
                _ => panic!("{}", self.debug_string),
            }
            self.program_counter += 2;
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let bytes = std::fs::read(args.get(1).unwrap()).unwrap();
    let (sender, reciever) = mpsc::channel();
    thread::spawn(move || Chip8::new(bytes, sender).run());
    create_chip8_window(reciever);
}

fn create_chip8_window(receiver: Receiver<[bool; WIDTH * HEIGHT]>) -> ! {
    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();

    let window = {
        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        let scaled_size = LogicalSize::new(
            WIDTH as f64 * VIEWPORT_MULTIPLYER,
            HEIGHT as f64 * VIEWPORT_MULTIPLYER,
        );
        WindowBuilder::new()
            .with_title("Chip8")
            .with_inner_size(scaled_size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(WIDTH as u32, HEIGHT as u32, surface_texture).unwrap()
    };

    event_loop.run(move |event, _, control_flow| {
        if let Event::RedrawRequested(_) = event {
            if let Err(err) = pixels.render() {
                eprintln!("{}", err);
                *control_flow = ControlFlow::Exit;
                return;
            }
        }
        let message = receiver.try_recv();
        if let Ok(msg) = message {
            render(pixels.frame_mut(), msg);
            window.request_redraw();
        }
        if input.update(&event) {
            // Close events
            if input.key_pressed(VirtualKeyCode::Escape) || input.close_requested() {
                *control_flow = ControlFlow::Exit;
                return;
            }

            CHIP_INPUT[1].store(input.key_held(VirtualKeyCode::Key1), Relaxed);
            CHIP_INPUT[2].store(input.key_held(VirtualKeyCode::Key2), Relaxed);
            CHIP_INPUT[3].store(input.key_held(VirtualKeyCode::Key3), Relaxed);
            CHIP_INPUT[12].store(input.key_held(VirtualKeyCode::Key4), Relaxed);
            CHIP_INPUT[4].store(input.key_held(VirtualKeyCode::Q), Relaxed);
            CHIP_INPUT[5].store(input.key_held(VirtualKeyCode::W), Relaxed);
            CHIP_INPUT[6].store(input.key_held(VirtualKeyCode::E), Relaxed);
            CHIP_INPUT[13].store(input.key_held(VirtualKeyCode::R), Relaxed);
            CHIP_INPUT[7].store(input.key_held(VirtualKeyCode::A), Relaxed);
            CHIP_INPUT[8].store(input.key_held(VirtualKeyCode::S), Relaxed);
            CHIP_INPUT[9].store(input.key_held(VirtualKeyCode::D), Relaxed);
            CHIP_INPUT[14].store(input.key_held(VirtualKeyCode::F), Relaxed);
            CHIP_INPUT[10].store(input.key_held(VirtualKeyCode::Z), Relaxed);
            CHIP_INPUT[0].store(input.key_held(VirtualKeyCode::X), Relaxed);
            CHIP_INPUT[11].store(input.key_held(VirtualKeyCode::C), Relaxed);
            CHIP_INPUT[15].store(input.key_held(VirtualKeyCode::V), Relaxed);
        }
    })
}

fn render(pixels: &mut [u8], bool_buffer: [bool; WIDTH * HEIGHT]) {
    for (i, pixel) in bool_buffer.iter().enumerate() {
        if *pixel {
            pixels[i * 4] = ON_COLOR_RGBA[0];
            pixels[i * 4 + 1] = ON_COLOR_RGBA[1];
            pixels[i * 4 + 2] = ON_COLOR_RGBA[2];
            pixels[i * 4 + 3] = ON_COLOR_RGBA[3];
        } else {
            pixels[i * 4] = OFF_COLOR_RGBA[0];
            pixels[i * 4 + 1] = OFF_COLOR_RGBA[1];
            pixels[i * 4 + 2] = OFF_COLOR_RGBA[2];
            pixels[i * 4 + 3] = OFF_COLOR_RGBA[3];
        }
    }
}
