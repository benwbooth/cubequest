use macroquad::prelude::*;
use rustysynth::{SoundFont, Synthesizer, SynthesizerSettings};
use std::fs::File;
use std::io::BufReader;
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
// AUTOMIXING HELPERS
// =============================================================================

/// Calculate RMS level of audio buffer
fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() { return 0.0; }
    let sum: f32 = samples.iter().map(|s| s * s).sum();
    (sum / samples.len() as f32).sqrt()
}

/// Soft limiter to prevent harsh clipping
fn soft_limit(sample: f32) -> f32 {
    if sample.abs() > 0.9 {
        let excess = sample.abs() - 0.9;
        let limited = 0.9 + excess.tanh() * 0.1;
        limited.copysign(sample)
    } else {
        sample
    }
}

/// Normalize channel to target RMS with soft limiting
fn normalize_channel(samples: &mut [f32], target_rms: f32) {
    let rms = calculate_rms(samples);
    if rms < 0.0001 { return; } // Skip silent channels

    let gain = target_rms / rms;
    for sample in samples.iter_mut() {
        *sample = soft_limit(*sample * gain);
    }
}

// =============================================================================
// MIDI MUSIC SYSTEM
// =============================================================================

fn load_music_samples(soundfont_path: &str) -> Option<Vec<f32>> {
    use music::dsp::Effect;
    use music::{MidiEventType, extract_channel_effects, channels_with_effects};
    use std::collections::HashMap;

    let sf2_file = File::open(soundfont_path).ok()?;
    let soundfont = Arc::new(SoundFont::new(&mut BufReader::new(sf2_file)).ok()?);

    // Use the music DSL to compose dungeon music
    let composition = compositions::dungeon_ambient();

    // Get individual MIDI events
    let events = note_to_events(&composition, 128.0); // tempo 128 BPM

    // PASS 1: Extract effect configurations from composition
    let effect_configs = extract_channel_effects(&events);
    let effect_channels = channels_with_effects(&events);

    eprintln!("Generated {} MIDI events, {} channels with effects: {:?}",
        events.len(), effect_channels.len(), effect_channels);

    // Find total duration
    let total_duration = events.iter()
        .map(|e| e.time)
        .fold(0.0_f32, |a, b| a.max(b)) + 1.0;

    let total_samples = (total_duration * SAMPLE_RATE as f32) as usize;

    // Create synthesizers: one for main (no effects), one per effect channel
    let settings = SynthesizerSettings::new(SAMPLE_RATE as i32);
    let mut synth_main = Synthesizer::new(&soundfont, &settings).ok()?;

    // Create synths and buffers for each channel with effects
    let mut effect_synths: HashMap<u8, Synthesizer> = HashMap::new();
    let mut effect_buffers_l: HashMap<u8, Vec<f32>> = HashMap::new();
    let mut effect_buffers_r: HashMap<u8, Vec<f32>> = HashMap::new();

    for &channel in &effect_channels {
        effect_synths.insert(channel, Synthesizer::new(&soundfont, &settings).ok()?);
        effect_buffers_l.insert(channel, vec![0.0f32; total_samples]);
        effect_buffers_r.insert(channel, vec![0.0f32; total_samples]);
    }

    // Sort events by time
    let mut sorted_events = events.clone();
    sorted_events.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

    // PASS 2: Render - route events to appropriate synths
    let samples_per_block = 64;
    let mut main_left = vec![0.0f32; total_samples];
    let mut main_right = vec![0.0f32; total_samples];

    let mut event_idx = 0;
    let mut sample_pos = 0;

    while sample_pos < total_samples {
        let current_time = sample_pos as f32 / SAMPLE_RATE as f32;

        // Process all events up to current time
        while event_idx < sorted_events.len() && sorted_events[event_idx].time <= current_time {
            let event = &sorted_events[event_idx];
            let has_effects = effect_channels.contains(&event.channel);

            // Route to effect synth or main synth
            let (synth, channel) = if has_effects {
                (effect_synths.get_mut(&event.channel).unwrap(), 0i32)
            } else {
                (&mut synth_main, event.channel as i32)
            };

            match &event.event_type {
                MidiEventType::NoteOn { note, velocity } => {
                    synth.note_on(channel, *note as i32, *velocity as i32);
                }
                MidiEventType::NoteOff { note } => {
                    synth.note_off(channel, *note as i32);
                }
                MidiEventType::ProgramChange { program } => {
                    synth.process_midi_message(channel, 0xC0, *program as i32, 0);
                }
                MidiEventType::ControlChange { controller, value } => {
                    synth.process_midi_message(channel, 0xB0, *controller as i32, *value as i32);
                }
                MidiEventType::PitchBend { value } => {
                    let lsb = (*value & 0x7F) as i32;
                    let msb = ((*value >> 7) & 0x7F) as i32;
                    synth.process_midi_message(channel, 0xE0, lsb, msb);
                }
                MidiEventType::Comment(_) | MidiEventType::ChannelEffects { .. } => {}
            }
            event_idx += 1;
        }

        // Render a block from each synth
        let block_end = (sample_pos + samples_per_block).min(total_samples);
        let block_size = block_end - sample_pos;

        // Render main synth
        let mut block_main_l = vec![0.0f32; block_size];
        let mut block_main_r = vec![0.0f32; block_size];
        synth_main.render(&mut block_main_l, &mut block_main_r);

        for i in 0..block_size {
            main_left[sample_pos + i] = block_main_l[i];
            main_right[sample_pos + i] = block_main_r[i];
        }

        // Render each effect synth
        for &channel in &effect_channels {
            let synth = effect_synths.get_mut(&channel).unwrap();
            let mut block_l = vec![0.0f32; block_size];
            let mut block_r = vec![0.0f32; block_size];
            synth.render(&mut block_l, &mut block_r);

            let buf_l = effect_buffers_l.get_mut(&channel).unwrap();
            let buf_r = effect_buffers_r.get_mut(&channel).unwrap();
            for i in 0..block_size {
                buf_l[sample_pos + i] = block_l[i];
                buf_r[sample_pos + i] = block_r[i];
            }
        }

        sample_pos = block_end;
    }

    // PASS 3: Apply DSP effects from composition to each effect channel
    for (&channel, configs) in &effect_configs {
        eprintln!("Applying {} effects to channel {}", configs.len(), channel);

        // Create effect chains from configs
        let mut chain_l: Vec<Box<dyn Effect>> = configs.iter().map(|c| c.create()).collect();
        let mut chain_r: Vec<Box<dyn Effect>> = configs.iter().map(|c| c.create()).collect();

        // Apply effects
        let buf_l = effect_buffers_l.get_mut(&channel).unwrap();
        let buf_r = effect_buffers_r.get_mut(&channel).unwrap();

        for sample in buf_l.iter_mut() {
            for effect in &mut chain_l {
                *sample = effect.process(*sample);
            }
        }
        for sample in buf_r.iter_mut() {
            for effect in &mut chain_r {
                *sample = effect.process(*sample);
            }
        }
    }

    // PASS 4: Automixing - normalize effect channels to target RMS
    // Target RMS levels per channel (tuned for good balance)
    let target_rms: std::collections::HashMap<u8, f32> = [
        (3, 0.07),  // Guitar (ch 3) - distorted, sits behind main mix
    ].into();

    for (&channel, buf_l) in effect_buffers_l.iter_mut() {
        let target = target_rms.get(&channel).copied().unwrap_or(0.08);
        let rms_before = calculate_rms(buf_l);
        normalize_channel(buf_l, target);
        eprintln!("Ch{} L: RMS {:.4} -> {:.4}", channel, rms_before, calculate_rms(buf_l));
    }
    for (&channel, buf_r) in effect_buffers_r.iter_mut() {
        let target = target_rms.get(&channel).copied().unwrap_or(0.08);
        normalize_channel(buf_r, target);
    }

    // Normalize main mix
    let main_target_rms = 0.12;
    normalize_channel(&mut main_left, main_target_rms);
    normalize_channel(&mut main_right, main_target_rms);
    eprintln!("Main mix normalized to RMS {:.4}", calculate_rms(&main_left));

    // Mix all channels together (volumes now balanced via RMS normalization)
    let mono: Vec<f32> = (0..total_samples)
        .map(|i| {
            let mut l = main_left[i];
            let mut r = main_right[i];

            // Add processed effect channels
            for buf_l in effect_buffers_l.values() {
                l += buf_l[i];
            }
            for buf_r in effect_buffers_r.values() {
                r += buf_r[i];
            }

            soft_limit((l + r) * 0.5)
        })
        .collect();

    eprintln!("Music rendered: {} samples ({:.1}s) with DSP effects from composition",
        mono.len(), mono.len() as f32 / SAMPLE_RATE as f32);
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

/// Animation state for an entity
#[derive(Clone)]
enum Animation {
    None,
    /// Walking from (from_x, from_y) to current position
    Walking { from_x: f32, from_y: f32, start_time: f64 },
    /// Attacking toward (target_x, target_y) - quick vicious lunge and back
    Attacking { target_x: f32, target_y: f32, start_time: f64 },
    /// Recoiling from a hit - knocked back then bounce back
    Recoiling { from_x: f32, from_y: f32, start_time: f64 },
}

#[derive(Clone)]
struct Entity {
    x: i32,
    y: i32,
    health: i32,
    max_health: i32,
    attack: i32,
    /// Current animation state
    anim: Animation,
}

impl Entity {
    fn new(x: i32, y: i32, health: i32, attack: i32) -> Self {
        Self {
            x,
            y,
            health,
            max_health: health,
            attack,
            anim: Animation::None,
        }
    }

    /// Get visual position (for rendering with animations)
    fn visual_pos(&self, current_time: f64) -> (f32, f32) {
        match &self.anim {
            Animation::None => (self.x as f32, self.y as f32),
            Animation::Walking { from_x, from_y, start_time } => {
                let elapsed = current_time - start_time;
                let t = (elapsed / WALK_ANIM_DURATION).min(1.0) as f32;
                // Ease out
                let t = 1.0 - (1.0 - t) * (1.0 - t);
                let x = from_x + (self.x as f32 - from_x) * t;
                let y = from_y + (self.y as f32 - from_y) * t;
                (x, y)
            }
            Animation::Attacking { target_x, target_y, start_time } => {
                let elapsed = current_time - start_time;
                let total_duration = ATTACK_ANIM_DURATION;
                let t = (elapsed / total_duration).min(1.0) as f32;

                // Quick vicious lunge: fast forward (20%), hold at peak (10%), snap back (70%)
                let lunge_amount = if t < 0.2 {
                    // Fast lunge forward - ease out for snap
                    let phase = t / 0.2;
                    let eased = 1.0 - (1.0 - phase).powi(3); // Cubic ease out
                    eased * 0.7 // Go 70% of the way
                } else if t < 0.3 {
                    // Hold at peak briefly
                    0.7
                } else {
                    // Snap back quickly
                    let phase = (t - 0.3) / 0.7;
                    let eased = phase.powi(2); // Ease in for weight
                    0.7 * (1.0 - eased)
                };

                let base_x = self.x as f32;
                let base_y = self.y as f32;
                let dx = target_x - base_x;
                let dy = target_y - base_y;

                (base_x + dx * lunge_amount, base_y + dy * lunge_amount)
            }
            Animation::Recoiling { from_x, from_y, start_time } => {
                let elapsed = current_time - start_time;
                let total_duration = RECOIL_ANIM_DURATION;
                let t = (elapsed / total_duration).min(1.0) as f32;

                // Knocked back then bounce: fly back (30%), bounce back (70%)
                let base_x = self.x as f32;
                let base_y = self.y as f32;
                // Direction away from attacker
                let dx = base_x - from_x;
                let dy = base_y - from_y;
                // Normalize
                let len = (dx * dx + dy * dy).sqrt().max(0.001);
                let nx = dx / len;
                let ny = dy / len;

                let knockback = if t < 0.3 {
                    // Fly back - ease out
                    let phase = t / 0.3;
                    let eased = 1.0 - (1.0 - phase).powi(2);
                    eased * 0.4 // Knockback 40% of a tile
                } else {
                    // Bounce back with overshoot
                    let phase = (t - 0.3) / 0.7;
                    // Damped oscillation
                    let bounce = (1.0 - phase) * (phase * 3.14159 * 1.5).sin() * 0.3;
                    0.4 * (1.0 - phase) + bounce
                };

                (base_x + nx * knockback, base_y + ny * knockback)
            }
        }
    }

    /// Check if animation is complete
    fn is_anim_done(&self, current_time: f64) -> bool {
        match &self.anim {
            Animation::None => true,
            Animation::Walking { start_time, .. } => {
                current_time - start_time >= WALK_ANIM_DURATION
            }
            Animation::Attacking { start_time, .. } => {
                current_time - start_time >= ATTACK_ANIM_DURATION
            }
            Animation::Recoiling { start_time, .. } => {
                current_time - start_time >= RECOIL_ANIM_DURATION
            }
        }
    }

    /// Clear animation if done
    fn update_anim(&mut self, current_time: f64) {
        if self.is_anim_done(current_time) {
            self.anim = Animation::None;
        }
    }

    /// Start walking animation
    fn start_walk(&mut self, from_x: i32, from_y: i32, current_time: f64) {
        self.anim = Animation::Walking {
            from_x: from_x as f32,
            from_y: from_y as f32,
            start_time: current_time,
        };
    }

    /// Start attack animation
    fn start_attack(&mut self, target_x: i32, target_y: i32, current_time: f64) {
        self.anim = Animation::Attacking {
            target_x: target_x as f32,
            target_y: target_y as f32,
            start_time: current_time,
        };
    }

    /// Start recoil animation (knocked back from attacker position)
    fn start_recoil(&mut self, attacker_x: i32, attacker_y: i32, current_time: f64) {
        self.anim = Animation::Recoiling {
            from_x: attacker_x as f32,
            from_y: attacker_y as f32,
            start_time: current_time,
        };
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

/// Pending combat action
#[derive(Clone)]
enum CombatAction {
    PlayerAttack { enemy_idx: usize, damage: i32 },
    EnemyAttack { enemy_idx: usize, damage: i32 },
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
    /// Pending combat actions to execute in sequence
    combat_queue: Vec<CombatAction>,
    /// Time when current combat action started
    combat_action_start: f64,
    /// Is the game waiting for animations to complete?
    waiting_for_anim: bool,
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
            combat_queue: Vec::new(),
            combat_action_start: 0.0,
            waiting_for_anim: false,
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
        if self.game_over || self.waiting_for_anim {
            return;
        }

        let current_time = get_time();
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

        if let Some(enemy_idx) = enemy_hit {
            // Start combat sequence - player attacks first
            let damage = self.player.attack;
            self.combat_queue.clear();
            self.combat_queue.push(CombatAction::PlayerAttack { enemy_idx, damage });

            // Check if enemy will counter-attack (if it survives)
            let enemy = &self.enemies[enemy_idx];
            if enemy.health > damage {
                // Enemy will survive and counter-attack
                let enemy_damage = enemy.attack;
                self.combat_queue.push(CombatAction::EnemyAttack { enemy_idx, damage: enemy_damage });
            }

            // Start player attack animation
            self.player.start_attack(new_x, new_y, current_time);
            self.combat_action_start = current_time;
            self.waiting_for_anim = true;
            self.play_sfx(SfxId::Hit);
        } else {
            // Move with animation
            let old_x = self.player.x;
            let old_y = self.player.y;
            self.player.x = new_x;
            self.player.y = new_y;
            self.player.start_walk(old_x, old_y, current_time);
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
            } else {
                // Enemy turns (only if not taking stairs)
                self.enemy_turn();
            }
        }
    }

    /// Update combat animations and process combat queue
    fn update(&mut self) {
        let current_time = get_time();

        // Update player animation
        self.player.update_anim(current_time);

        // Update enemy animations
        for enemy in &mut self.enemies {
            enemy.update_anim(current_time);
        }

        if !self.waiting_for_anim {
            return;
        }

        // Check if current animation is done
        let anim_done = self.player.is_anim_done(current_time)
            && self.enemies.iter().all(|e| e.is_anim_done(current_time));

        if !anim_done {
            return;
        }

        // Process next combat action
        if let Some(action) = self.combat_queue.first().cloned() {
            self.combat_queue.remove(0);

            match action {
                CombatAction::PlayerAttack { enemy_idx, damage } => {
                    // Apply damage and start enemy recoil
                    if enemy_idx < self.enemies.len() {
                        self.enemies[enemy_idx].health -= damage;
                        // Enemy recoils from player's position
                        self.enemies[enemy_idx].start_recoil(self.player.x, self.player.y, current_time);
                        self.add_message(&format!("You hit the enemy for {} damage!", damage));

                        if self.enemies[enemy_idx].health <= 0 {
                            self.enemies.remove(enemy_idx);
                            self.score += 50;
                            self.add_message("Enemy defeated!");
                            self.play_sfx(SfxId::EnemyDie);
                            // Clear remaining combat actions since enemy is dead
                            self.combat_queue.clear();
                        }
                    }

                    // If there are more actions, start next animation after recoil
                    if let Some(next_action) = self.combat_queue.first() {
                        match next_action {
                            CombatAction::EnemyAttack { enemy_idx, .. } => {
                                if *enemy_idx < self.enemies.len() {
                                    let target_x = self.player.x;
                                    let target_y = self.player.y;
                                    self.enemies[*enemy_idx].start_attack(target_x, target_y, current_time);
                                    self.combat_action_start = current_time;
                                    self.play_sfx(SfxId::Hit);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                CombatAction::EnemyAttack { enemy_idx, damage } => {
                    // Apply damage to player and start player recoil
                    self.player.health -= damage;
                    // Player recoils from enemy's position
                    if enemy_idx < self.enemies.len() {
                        self.player.start_recoil(self.enemies[enemy_idx].x, self.enemies[enemy_idx].y, current_time);
                    }
                    self.add_message(&format!("Enemy hits you for {} damage!", damage));
                    self.play_sfx(SfxId::Hurt);

                    if self.player.health <= 0 {
                        self.game_over = true;
                        self.add_message("You died! Game Over.");
                    }
                }
            }
        }

        // Check if combat is done (player and all enemies done animating)
        let enemies_done = self.enemies.iter().all(|e| e.is_anim_done(current_time));
        if self.combat_queue.is_empty() && self.player.is_anim_done(current_time) && enemies_done {
            self.waiting_for_anim = false;

            // Now do enemy turn (movement only, not attacks)
            if !self.game_over {
                self.enemy_turn();
            }
        }
    }

    fn enemy_turn(&mut self) {
        let current_time = get_time();
        let player_x = self.player.x;
        let player_y = self.player.y;

        for i in 0..self.enemies.len() {
            let enemy_x = self.enemies[i].x;
            let enemy_y = self.enemies[i].y;

            // Simple chase AI
            let dx = (player_x - enemy_x).signum();
            let dy = (player_y - enemy_y).signum();

            // Try to move towards player (but not into player - combat happens on player's turn)
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

                // Don't move into player (combat happens on player's bump)
                if new_x == player_x && new_y == player_y {
                    continue;
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
                        let old_x = self.enemies[i].x;
                        let old_y = self.enemies[i].y;
                        self.enemies[i].x = new_x;
                        self.enemies[i].y = new_y;
                        self.enemies[i].start_walk(old_x, old_y, current_time);
                        break;
                    }
                }
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
        self.combat_queue.clear();
        self.waiting_for_anim = false;
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
        let current_time = get_time();
        for enemy in &self.enemies {
            let (vx, vy) = enemy.visual_pos(current_time);
            let screen_x = offset_x + vx * TILE_SIZE * SCALE;
            let screen_y = offset_y + vy * TILE_SIZE * SCALE;
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
        let (pvx, pvy) = self.player.visual_pos(current_time);
        let player_screen_x = offset_x + pvx * TILE_SIZE * SCALE;
        let player_screen_y = offset_y + pvy * TILE_SIZE * SCALE;
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
const WALK_ANIM_DURATION: f64 = 0.08; // Smooth walk animation duration
const ATTACK_ANIM_DURATION: f64 = 0.12; // Quick vicious attack
const RECOIL_ANIM_DURATION: f64 = 0.25; // Knockback and bounce

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

        // Update animations and combat
        game.update();

        // Draw
        game.draw();

        next_frame().await
    }
}
