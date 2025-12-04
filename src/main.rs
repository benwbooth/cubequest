use macroquad::prelude::*;
use rustysynth::{MidiFile, MidiFileSequencer, SoundFont, Synthesizer, SynthesizerSettings};
use std::fs::File;
use std::io::{BufReader, Cursor};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use music::prelude::*;
use music::compositions;
use music::sfx::SfxLibrary as MusicSfxLibrary;

// Use macroquad's built-in rand
fn rand_range(min: i32, max: i32) -> i32 {
    ::macroquad::rand::gen_range(min, max)
}

fn rand_f32() -> f32 {
    ::macroquad::rand::gen_range(0.0, 1.0)
}

fn rand_bool(probability: f64) -> bool {
    rand_f32() < probability as f32
}

// =============================================================================
// CONSTANTS
// =============================================================================

const TILE_SIZE: f32 = 16.0;
const SCALE: f32 = 3.0;
const MAP_WIDTH: usize = 40;
const MAP_HEIGHT: usize = 30;
const PIXEL_SIZE: f32 = SCALE * 2.0; // Each sprite pixel rendered at this size

// =============================================================================
// COLORS (Palette)
// =============================================================================

const C_BLACK: Color = Color::new(0.1, 0.1, 0.12, 1.0);
const C_DARK: Color = Color::new(0.2, 0.18, 0.25, 1.0);
const C_WALL: Color = Color::new(0.35, 0.3, 0.4, 1.0);
const C_FLOOR: Color = Color::new(0.25, 0.22, 0.3, 1.0);
const C_PLAYER_SKIN: Color = Color::new(0.9, 0.75, 0.6, 1.0);
const C_PLAYER_HAIR: Color = Color::new(0.6, 0.4, 0.2, 1.0);
const C_PLAYER_SHIRT: Color = Color::new(0.3, 0.5, 0.8, 1.0);
const C_ENEMY_BODY: Color = Color::new(0.7, 0.2, 0.2, 1.0);
const C_ENEMY_EYE: Color = Color::new(1.0, 1.0, 0.3, 1.0);
const C_GOLD: Color = Color::new(1.0, 0.85, 0.2, 1.0);
const C_HEALTH: Color = Color::new(0.9, 0.2, 0.3, 1.0);
const C_STAIRS: Color = Color::new(0.4, 0.8, 0.5, 1.0);

// =============================================================================
// PIXEL SPRITES (8x8)
// 0 = transparent, 1-9 = palette indices
// =============================================================================

const SPRITE_PLAYER: [[u8; 8]; 8] = [
    [0, 0, 2, 2, 2, 2, 0, 0],
    [0, 2, 2, 2, 2, 2, 2, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 3, 3, 3, 3, 0, 0],
    [0, 0, 3, 3, 3, 3, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
];

const SPRITE_ENEMY: [[u8; 8]; 8] = [
    [0, 0, 4, 4, 4, 4, 0, 0],
    [0, 4, 4, 4, 4, 4, 4, 0],
    [4, 4, 5, 4, 4, 5, 4, 4],
    [4, 4, 4, 4, 4, 4, 4, 4],
    [4, 4, 4, 4, 4, 4, 4, 4],
    [0, 4, 4, 0, 0, 4, 4, 0],
    [0, 4, 4, 0, 0, 4, 4, 0],
    [0, 0, 4, 0, 0, 4, 0, 0],
];

const SPRITE_GOLD: [[u8; 8]; 8] = [
    [0, 0, 0, 6, 6, 0, 0, 0],
    [0, 0, 6, 6, 6, 6, 0, 0],
    [0, 6, 6, 9, 6, 6, 6, 0],
    [0, 6, 6, 6, 6, 6, 6, 0],
    [0, 6, 6, 6, 6, 6, 6, 0],
    [0, 6, 6, 6, 9, 6, 6, 0],
    [0, 0, 6, 6, 6, 6, 0, 0],
    [0, 0, 0, 6, 6, 0, 0, 0],
];

const SPRITE_HEALTH: [[u8; 8]; 8] = [
    [0, 7, 7, 0, 0, 7, 7, 0],
    [7, 7, 7, 7, 7, 7, 7, 7],
    [7, 7, 7, 7, 7, 7, 7, 7],
    [7, 7, 7, 7, 7, 7, 7, 7],
    [0, 7, 7, 7, 7, 7, 7, 0],
    [0, 0, 7, 7, 7, 7, 0, 0],
    [0, 0, 0, 7, 7, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
];

const SPRITE_STAIRS: [[u8; 8]; 8] = [
    [8, 8, 8, 8, 8, 8, 8, 8],
    [8, 0, 0, 0, 0, 0, 0, 8],
    [8, 8, 8, 8, 8, 8, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 8],
    [8, 0, 8, 8, 8, 8, 8, 8],
    [8, 0, 0, 0, 0, 0, 0, 8],
    [8, 8, 8, 8, 8, 8, 0, 8],
    [8, 8, 8, 8, 8, 8, 8, 8],
];

fn get_palette_color(index: u8) -> Option<Color> {
    match index {
        0 => None, // Transparent
        1 => Some(C_PLAYER_SKIN),
        2 => Some(C_PLAYER_HAIR),
        3 => Some(C_PLAYER_SHIRT),
        4 => Some(C_ENEMY_BODY),
        5 => Some(C_ENEMY_EYE),
        6 => Some(C_GOLD),
        7 => Some(C_HEALTH),
        8 => Some(C_STAIRS),
        9 => Some(Color::new(1.0, 1.0, 0.9, 1.0)), // Highlight
        _ => None,
    }
}

fn draw_sprite(sprite: &[[u8; 8]; 8], x: f32, y: f32) {
    for (row_idx, row) in sprite.iter().enumerate() {
        for (col_idx, &pixel) in row.iter().enumerate() {
            if let Some(color) = get_palette_color(pixel) {
                draw_rectangle(
                    x + col_idx as f32 * PIXEL_SIZE,
                    y + row_idx as f32 * PIXEL_SIZE,
                    PIXEL_SIZE,
                    PIXEL_SIZE,
                    color,
                );
            }
        }
    }
}

// =============================================================================
// PROCEDURAL SFX (Square, Triangle, Noise) - Using cpal for low latency
// =============================================================================

const SAMPLE_RATE: u32 = 44100;

// Maximum number of sounds playing simultaneously
const MAX_VOICES: usize = 8;

// Sound effect identifiers
#[derive(Clone, Copy)]
enum SfxId {
    Hit = 0,
    Pickup = 1,
    Hurt = 2,
    Step = 3,
    EnemyDie = 4,
    Stairs = 5,
}

// Lock-free SFX slot
struct SfxSlot {
    sound_id: AtomicUsize,  // Which sound to play (usize::MAX = none)
    position: AtomicUsize,
}

// Audio state shared between threads
struct AudioState {
    music_samples: Vec<f32>,
    music_pos: AtomicUsize,
    has_music: bool,
    sfx_slots: [SfxSlot; MAX_VOICES],
    sounds: [Vec<f32>; 6],
}

struct SfxLibrary {
    state: Arc<AudioState>,
    next_slot: AtomicUsize,
    _stream: cpal::Stream,
}

impl SfxLibrary {
    fn new(music_samples: Option<Vec<f32>>) -> Self {
        // Use the music crate's synthesizer-based SFX generation
        let music_sfx = MusicSfxLibrary::new();
        let sounds = [
            music_sfx.hit,
            music_sfx.pickup,
            music_sfx.hurt,
            music_sfx.step,
            music_sfx.enemy_die,
            music_sfx.stairs,
        ];

        let state = Arc::new(AudioState {
            music_samples: music_samples.clone().unwrap_or_default(),
            music_pos: AtomicUsize::new(0),
            has_music: music_samples.is_some(),
            sfx_slots: std::array::from_fn(|_| SfxSlot {
                sound_id: AtomicUsize::new(usize::MAX),
                position: AtomicUsize::new(0),
            }),
            sounds,
        });
        let state_clone = Arc::clone(&state);

        // Set up cpal
        let host = cpal::default_host();
        let device = host.default_output_device().expect("No output device");

        let config = cpal::StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(SAMPLE_RATE),
            buffer_size: cpal::BufferSize::Fixed(512), // ~11ms - stable with low latency
        };

        let stream = device.build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                // Clear buffer
                for sample in data.iter_mut() {
                    *sample = 0.0;
                }

                // Mix music
                if state_clone.has_music {
                    let mut pos = state_clone.music_pos.load(Ordering::Relaxed);
                    let music = &state_clone.music_samples;
                    for out in data.iter_mut() {
                        *out += music[pos] * 0.5;
                        pos += 1;
                        if pos >= music.len() {
                            pos = 0;
                        }
                    }
                    state_clone.music_pos.store(pos, Ordering::Relaxed);
                }

                // Mix SFX
                for slot in &state_clone.sfx_slots {
                    let sound_id = slot.sound_id.load(Ordering::Relaxed);
                    if sound_id >= 6 {
                        continue;
                    }
                    let samples = &state_clone.sounds[sound_id];
                    let mut pos = slot.position.load(Ordering::Relaxed);
                    for out in data.iter_mut() {
                        if pos < samples.len() {
                            *out += samples[pos];
                            pos += 1;
                        } else {
                            slot.sound_id.store(usize::MAX, Ordering::Relaxed);
                            break;
                        }
                    }
                    slot.position.store(pos, Ordering::Relaxed);
                }
            },
            |err| eprintln!("Audio error: {}", err),
            None,
        ).expect("Failed to build audio stream");

        stream.play().expect("Failed to start audio stream");
        eprintln!("Audio initialized (512 sample buffer = {:.1}ms)",
            512.0 / SAMPLE_RATE as f32 * 1000.0);

        Self {
            state,
            next_slot: AtomicUsize::new(0),
            _stream: stream,
        }
    }

    fn play(&self, id: SfxId) {
        let slot_idx = self.next_slot.fetch_add(1, Ordering::Relaxed) % MAX_VOICES;
        let slot = &self.state.sfx_slots[slot_idx];
        slot.position.store(0, Ordering::Relaxed);
        slot.sound_id.store(id as usize, Ordering::Release);
    }

}

// =============================================================================
// MIDI MUSIC SYSTEM
// =============================================================================

fn load_music_samples(soundfont_path: &str) -> Option<Vec<f32>> {
    let sf2_file = File::open(soundfont_path).ok()?;
    let soundfont = Arc::new(SoundFont::new(&mut BufReader::new(sf2_file)).ok()?);

    // Use the music DSL to compose dungeon music
    let composition = compositions::dungeon_ambient();
    let midi_data = compose_to_midi(&composition, 70.0);

    eprintln!("Generated MIDI: {} bytes from DSL composition", midi_data.len());

    let midi_file = MidiFile::new(&mut Cursor::new(midi_data)).ok()?;

    let settings = SynthesizerSettings::new(SAMPLE_RATE as i32);
    let synthesizer = Synthesizer::new(&soundfont, &settings).ok()?;
    let mut sequencer = MidiFileSequencer::new(synthesizer);

    let midi_file = Arc::new(midi_file);
    sequencer.play(&midi_file, true);

    // Render enough for the composition to loop nicely
    let duration_samples = SAMPLE_RATE * 60; // 60 seconds
    let mut left = vec![0f32; duration_samples as usize];
    let mut right = vec![0f32; duration_samples as usize];
    sequencer.render(&mut left, &mut right);

    // Mix to mono with volume boost
    let volume = 1.5;
    let mono: Vec<f32> = left.iter().zip(right.iter())
        .map(|(l, r)| ((l + r) * 0.5 * volume).clamp(-1.0, 1.0))
        .collect();

    eprintln!("Music rendered: {} samples ({:.1}s)", mono.len(), mono.len() as f32 / SAMPLE_RATE as f32);
    Some(mono)
}


// =============================================================================
// GAME STRUCTURES
// =============================================================================

#[derive(Clone, Copy, PartialEq)]
enum Tile {
    Wall,
    Floor,
    Stairs,
}

#[derive(Clone)]
struct Entity {
    x: i32,
    y: i32,
    health: i32,
    max_health: i32,
    attack: i32,
}

impl Entity {
    fn new(x: i32, y: i32, health: i32, attack: i32) -> Self {
        Self {
            x,
            y,
            health,
            max_health: health,
            attack,
        }
    }
}

struct Item {
    x: i32,
    y: i32,
    kind: ItemKind,
}

#[derive(Clone, Copy)]
enum ItemKind {
    Gold(i32),
    Health(i32),
}

struct Room {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
}

impl Room {
    fn center(&self) -> (i32, i32) {
        (self.x + self.width / 2, self.y + self.height / 2)
    }

    fn intersects(&self, other: &Room) -> bool {
        self.x <= other.x + other.width
            && self.x + self.width >= other.x
            && self.y <= other.y + other.height
            && self.y + self.height >= other.y
    }
}

struct Game {
    map: [[Tile; MAP_HEIGHT]; MAP_WIDTH],
    player: Entity,
    enemies: Vec<Entity>,
    items: Vec<Item>,
    floor: i32,
    score: i32,
    messages: Vec<(String, f64)>,
    game_over: bool,
    won: bool,
    sfx: SfxLibrary,
    last_move_time: f64,
}

impl Game {
    async fn new() -> Self {
        // Try to load music
        let music_samples = load_music_samples("SC55_zzdenis_v0.5.sf2");
        if music_samples.is_none() {
            eprintln!("Failed to load music - check SC55_zzdenis_v0.5.sf2 exists");
        }

        // Create SFX library with music
        let sfx = SfxLibrary::new(music_samples);

        let mut game = Self {
            map: [[Tile::Wall; MAP_HEIGHT]; MAP_WIDTH],
            player: Entity::new(0, 0, 10, 3),
            enemies: Vec::new(),
            items: Vec::new(),
            floor: 1,
            score: 0,
            messages: Vec::new(),
            game_over: false,
            won: false,
            sfx,
            last_move_time: 0.0,
        };
        game.generate_floor();
        game.add_message("Welcome to the dungeon! Reach floor 5 to win.");
        game
    }

    fn play_sfx(&self, id: SfxId) {
        self.sfx.play(id);
    }

    fn add_message(&mut self, msg: &str) {
        self.messages.push((msg.to_string(), get_time()));
    }

    fn generate_floor(&mut self) {
        // Reset map
        for x in 0..MAP_WIDTH {
            for y in 0..MAP_HEIGHT {
                self.map[x][y] = Tile::Wall;
            }
        }
        self.enemies.clear();
        self.items.clear();

        let mut rooms = Vec::new();
        let num_rooms = 6 + self.floor as usize;

        // Generate rooms
        for _ in 0..50 {
            if rooms.len() >= num_rooms {
                break;
            }
            let w = rand_range(4, 10);
            let h = rand_range(4, 8);
            let x = rand_range(1, MAP_WIDTH as i32 - w - 1);
            let y = rand_range(1, MAP_HEIGHT as i32 - h - 1);
            let room = Room {
                x,
                y,
                width: w,
                height: h,
            };

            let mut ok = true;
            for other in &rooms {
                if room.intersects(other) {
                    ok = false;
                    break;
                }
            }
            if ok {
                rooms.push(room);
            }
        }

        // Carve rooms
        for room in &rooms {
            for x in room.x..(room.x + room.width) {
                for y in room.y..(room.y + room.height) {
                    self.map[x as usize][y as usize] = Tile::Floor;
                }
            }
        }

        // Connect rooms with corridors
        for i in 1..rooms.len() {
            let (x1, y1) = rooms[i - 1].center();
            let (x2, y2) = rooms[i].center();

            // Horizontal then vertical
            let (start_x, end_x) = if x1 < x2 { (x1, x2) } else { (x2, x1) };
            for x in start_x..=end_x {
                self.map[x as usize][y1 as usize] = Tile::Floor;
            }
            let (start_y, end_y) = if y1 < y2 { (y1, y2) } else { (y2, y1) };
            for y in start_y..=end_y {
                self.map[x2 as usize][y as usize] = Tile::Floor;
            }
        }

        // Place player in first room
        let (px, py) = rooms[0].center();
        self.player.x = px;
        self.player.y = py;

        // Place stairs in last room
        let (sx, sy) = rooms[rooms.len() - 1].center();
        self.map[sx as usize][sy as usize] = Tile::Stairs;

        // Place enemies in other rooms
        let enemy_count = 2 + self.floor as usize;
        for i in 1..rooms.len() - 1 {
            if self.enemies.len() >= enemy_count {
                break;
            }
            let room = &rooms[i];
            let ex = rand_range(room.x + 1, room.x + room.width - 1);
            let ey = rand_range(room.y + 1, room.y + room.height - 1);
            let health = 3 + self.floor;
            let attack = 1 + self.floor / 2;
            self.enemies.push(Entity::new(ex, ey, health, attack));
        }

        // Place items
        for room in &rooms[1..rooms.len() - 1] {
            if rand_bool(0.6) {
                let ix = rand_range(room.x + 1, room.x + room.width - 1);
                let iy = rand_range(room.y + 1, room.y + room.height - 1);
                let kind = if rand_bool(0.3) {
                    ItemKind::Health(3)
                } else {
                    ItemKind::Gold(10 + rand_range(0, 10) * self.floor)
                };
                self.items.push(Item { x: ix, y: iy, kind });
            }
        }

        self.add_message(&format!("Floor {} - Find the stairs!", self.floor));
    }

    fn try_move(&mut self, dx: i32, dy: i32) {
        if self.game_over {
            return;
        }

        let new_x = self.player.x + dx;
        let new_y = self.player.y + dy;

        // Bounds check
        if new_x < 0 || new_x >= MAP_WIDTH as i32 || new_y < 0 || new_y >= MAP_HEIGHT as i32 {
            return;
        }

        // Wall check
        if self.map[new_x as usize][new_y as usize] == Tile::Wall {
            return;
        }

        // Enemy collision (combat)
        let mut enemy_hit = None;
        for (i, enemy) in self.enemies.iter().enumerate() {
            if enemy.x == new_x && enemy.y == new_y {
                enemy_hit = Some(i);
                break;
            }
        }

        if let Some(i) = enemy_hit {
            // Combat!
            let damage = self.player.attack;
            self.enemies[i].health -= damage;
            self.add_message(&format!("You hit the enemy for {} damage!", damage));
            self.play_sfx(SfxId::Hit);

            if self.enemies[i].health <= 0 {
                self.enemies.remove(i);
                self.score += 50;
                self.add_message("Enemy defeated!");
                self.play_sfx(SfxId::EnemyDie);
            }
        } else {
            // Move
            self.player.x = new_x;
            self.player.y = new_y;
            self.play_sfx(SfxId::Step);

            // Check items
            let mut picked_up = Vec::new();
            let mut pickup_messages = Vec::new();
            for (i, item) in self.items.iter().enumerate() {
                if item.x == new_x && item.y == new_y {
                    picked_up.push(i);
                    match item.kind {
                        ItemKind::Gold(amount) => {
                            self.score += amount;
                            pickup_messages.push(format!("Picked up {} gold!", amount));
                        }
                        ItemKind::Health(amount) => {
                            self.player.health =
                                (self.player.health + amount).min(self.player.max_health);
                            pickup_messages.push(format!("Restored {} health!", amount));
                        }
                    }
                }
            }
            for msg in pickup_messages {
                self.add_message(&msg);
                self.play_sfx(SfxId::Pickup);
            }
            for i in picked_up.into_iter().rev() {
                self.items.remove(i);
            }

            // Check stairs
            if self.map[new_x as usize][new_y as usize] == Tile::Stairs {
                self.floor += 1;
                self.play_sfx(SfxId::Stairs);
                if self.floor > 5 {
                    self.game_over = true;
                    self.won = true;
                    self.add_message("You escaped the dungeon! YOU WIN!");
                } else {
                    self.generate_floor();
                }
            }
        }

        // Enemy turns
        self.enemy_turn();
    }

    fn enemy_turn(&mut self) {
        let player_x = self.player.x;
        let player_y = self.player.y;
        let mut damage_to_player = 0;
        let mut enemy_attack_msg = None;

        for i in 0..self.enemies.len() {
            let enemy_x = self.enemies[i].x;
            let enemy_y = self.enemies[i].y;
            let enemy_attack = self.enemies[i].attack;

            // Simple chase AI
            let dx = (player_x - enemy_x).signum();
            let dy = (player_y - enemy_y).signum();

            // Try to move towards player
            let moves = if rand_bool(0.5) {
                [(dx, 0), (0, dy)]
            } else {
                [(0, dy), (dx, 0)]
            };

            for (mx, my) in moves {
                if mx == 0 && my == 0 {
                    continue;
                }
                let new_x = enemy_x + mx;
                let new_y = enemy_y + my;

                // Check if would hit player
                if new_x == player_x && new_y == player_y {
                    // Attack!
                    damage_to_player += enemy_attack;
                    enemy_attack_msg = Some(format!("Enemy hits you for {} damage!", enemy_attack));
                    break;
                }

                // Check valid move
                if new_x >= 0
                    && new_x < MAP_WIDTH as i32
                    && new_y >= 0
                    && new_y < MAP_HEIGHT as i32
                    && self.map[new_x as usize][new_y as usize] != Tile::Wall
                {
                    // Check not moving into another enemy
                    let blocked = self
                        .enemies
                        .iter()
                        .any(|e| e.x == new_x && e.y == new_y);
                    if !blocked {
                        self.enemies[i].x = new_x;
                        self.enemies[i].y = new_y;
                        break;
                    }
                }
            }
        }

        // Apply damage after the loop
        if damage_to_player > 0 {
            self.player.health -= damage_to_player;
            if let Some(msg) = enemy_attack_msg {
                self.add_message(&msg);
            }
            self.play_sfx(SfxId::Hurt);

            if self.player.health <= 0 {
                self.game_over = true;
                self.add_message("You died! Game Over.");
            }
        }
    }

    fn restart(&mut self) {
        self.player = Entity::new(0, 0, 10, 3);
        self.floor = 1;
        self.score = 0;
        self.game_over = false;
        self.won = false;
        self.messages.clear();
        self.generate_floor();
        self.add_message("Welcome to the dungeon! Reach floor 5 to win.");
    }

    fn draw(&self) {
        clear_background(C_BLACK);

        let offset_x = 20.0;
        let offset_y = 60.0;

        // Draw map
        for x in 0..MAP_WIDTH {
            for y in 0..MAP_HEIGHT {
                let screen_x = offset_x + x as f32 * TILE_SIZE * SCALE;
                let screen_y = offset_y + y as f32 * TILE_SIZE * SCALE;
                let color = match self.map[x][y] {
                    Tile::Wall => C_WALL,
                    Tile::Floor => C_FLOOR,
                    Tile::Stairs => C_FLOOR, // Draw floor under stairs
                };
                draw_rectangle(
                    screen_x,
                    screen_y,
                    TILE_SIZE * SCALE,
                    TILE_SIZE * SCALE,
                    color,
                );

                // Draw stairs sprite
                if self.map[x][y] == Tile::Stairs {
                    draw_sprite(&SPRITE_STAIRS, screen_x, screen_y);
                }
            }
        }

        // Draw items
        for item in &self.items {
            let screen_x = offset_x + item.x as f32 * TILE_SIZE * SCALE;
            let screen_y = offset_y + item.y as f32 * TILE_SIZE * SCALE;
            match item.kind {
                ItemKind::Gold(_) => draw_sprite(&SPRITE_GOLD, screen_x, screen_y),
                ItemKind::Health(_) => draw_sprite(&SPRITE_HEALTH, screen_x, screen_y),
            }
        }

        // Draw enemies
        for enemy in &self.enemies {
            let screen_x = offset_x + enemy.x as f32 * TILE_SIZE * SCALE;
            let screen_y = offset_y + enemy.y as f32 * TILE_SIZE * SCALE;
            draw_sprite(&SPRITE_ENEMY, screen_x, screen_y);

            // Health bar
            let bar_width = TILE_SIZE * SCALE;
            let bar_height = 4.0;
            let health_pct = enemy.health as f32 / enemy.max_health as f32;
            draw_rectangle(
                screen_x,
                screen_y - 6.0,
                bar_width,
                bar_height,
                C_DARK,
            );
            draw_rectangle(
                screen_x,
                screen_y - 6.0,
                bar_width * health_pct,
                bar_height,
                C_HEALTH,
            );
        }

        // Draw player
        let player_screen_x = offset_x + self.player.x as f32 * TILE_SIZE * SCALE;
        let player_screen_y = offset_y + self.player.y as f32 * TILE_SIZE * SCALE;
        draw_sprite(&SPRITE_PLAYER, player_screen_x, player_screen_y);

        // UI
        draw_text(
            &format!(
                "Floor: {}  HP: {}/{}  Score: {}",
                self.floor, self.player.health, self.player.max_health, self.score
            ),
            offset_x,
            30.0,
            30.0,
            WHITE,
        );

        // Messages (above the controls hint)
        let controls_height = 30.0;
        let msg_y = screen_height() - controls_height - 70.0;
        let current_time = get_time();
        for (i, (msg, time)) in self.messages.iter().rev().take(3).enumerate() {
            let age = current_time - time;
            let alpha = (1.0 - age / 5.0).max(0.0).min(1.0) as f32;
            draw_text(
                msg,
                offset_x,
                msg_y + i as f32 * 20.0,
                20.0,
                Color::new(1.0, 1.0, 1.0, alpha),
            );
        }

        // Game over overlay
        if self.game_over {
            let center_x = screen_width() / 2.0;
            let center_y = screen_height() / 2.0;
            draw_rectangle(
                center_x - 200.0,
                center_y - 60.0,
                400.0,
                120.0,
                Color::new(0.0, 0.0, 0.0, 0.8),
            );
            let text = if self.won { "YOU WIN!" } else { "GAME OVER" };
            let color = if self.won { C_GOLD } else { C_HEALTH };
            draw_text(text, center_x - 80.0, center_y - 10.0, 40.0, color);
            draw_text(
                &format!("Final Score: {}", self.score),
                center_x - 80.0,
                center_y + 20.0,
                24.0,
                WHITE,
            );
            draw_text(
                "Press R to restart",
                center_x - 70.0,
                center_y + 45.0,
                20.0,
                C_STAIRS,
            );
        }

        // Controls hint
        draw_text(
            "Arrow keys or WASD to move | R to restart",
            offset_x,
            screen_height() - 10.0,
            16.0,
            Color::new(0.5, 0.5, 0.5, 1.0),
        );
    }
}

// =============================================================================
// MAIN
// =============================================================================

fn window_conf() -> Conf {
    Conf {
        window_title: "Dungeon Depths".to_owned(),
        window_width: (MAP_WIDTH as f32 * TILE_SIZE * SCALE + 40.0) as i32,
        window_height: (MAP_HEIGHT as f32 * TILE_SIZE * SCALE + 140.0) as i32,
        ..Default::default()
    }
}

const MOVE_DELAY: f64 = 0.12; // Seconds between moves when holding key
const INITIAL_MOVE_DELAY: f64 = 0.0; // No delay on first press

#[macroquad::main(window_conf)]
async fn main() {
    let mut game = Game::new().await;
    let mut first_press = true;

    loop {
        let current_time = get_time();

        // Check if any movement key is pressed (not just initially pressed)
        let up = is_key_down(KeyCode::Up) || is_key_down(KeyCode::W);
        let down = is_key_down(KeyCode::Down) || is_key_down(KeyCode::S);
        let left = is_key_down(KeyCode::Left) || is_key_down(KeyCode::A);
        let right = is_key_down(KeyCode::Right) || is_key_down(KeyCode::D);

        // Check for fresh key press (for immediate response)
        let just_pressed = is_key_pressed(KeyCode::Up) || is_key_pressed(KeyCode::W)
            || is_key_pressed(KeyCode::Down) || is_key_pressed(KeyCode::S)
            || is_key_pressed(KeyCode::Left) || is_key_pressed(KeyCode::A)
            || is_key_pressed(KeyCode::Right) || is_key_pressed(KeyCode::D);

        if just_pressed {
            first_press = true;
        }

        let delay = if first_press { INITIAL_MOVE_DELAY } else { MOVE_DELAY };
        let can_move = current_time - game.last_move_time >= delay;

        if can_move && (up || down || left || right) {
            let (dx, dy) = if up {
                (0, -1)
            } else if down {
                (0, 1)
            } else if left {
                (-1, 0)
            } else {
                (1, 0)
            };
            game.try_move(dx, dy);
            game.last_move_time = current_time;
            first_press = false;
        }

        // Reset first_press when no movement keys are held
        if !up && !down && !left && !right {
            first_press = true;
        }

        if is_key_pressed(KeyCode::R) {
            game.restart();
        }
        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        // Draw
        game.draw();

        next_frame().await
    }
}
