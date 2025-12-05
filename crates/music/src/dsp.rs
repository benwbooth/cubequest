//! DSP Effects Module
//!
//! Stackable audio effects that can be applied per-channel.

/// Sample rate for DSP processing
pub const SAMPLE_RATE: f32 = 44100.0;

// =============================================================================
// EFFECT CONFIGURATION (for composition DSL)
// =============================================================================

/// Configuration for an effect - used in composition DSL
#[derive(Clone, Debug)]
pub enum EffectConfig {
    Distortion { drive: f32, mix: f32, output: f32 },
    Delay { time: f32, feedback: f32, mix: f32 },
    Chorus { rate: f32, depth: f32, mix: f32 },
    Flanger { rate: f32, depth: f32, feedback: f32, mix: f32 },
    Reverb { room_size: f32, damping: f32, mix: f32 },
    LowPass { cutoff: f32, resonance: f32 },
    HighPass { cutoff: f32, resonance: f32 },
    BandPass { cutoff: f32, resonance: f32 },
}

impl EffectConfig {
    /// Create the actual effect instance from config
    pub fn create(&self) -> Box<dyn Effect> {
        match self {
            EffectConfig::Distortion { drive, mix, output } => {
                Box::new(Distortion { drive: *drive, mix: *mix, output: *output, sample_count: 0, debug_printed: false })
            }
            EffectConfig::Delay { time, feedback, mix } => {
                Box::new(Delay::new(*time, *feedback, *mix))
            }
            EffectConfig::Chorus { rate, depth, mix } => {
                Box::new(Chorus::new(*rate, *depth, *mix))
            }
            EffectConfig::Flanger { rate, depth, feedback, mix } => {
                Box::new(Flanger::new(*rate, *depth, *feedback, *mix))
            }
            EffectConfig::Reverb { room_size, damping, mix } => {
                Box::new(Reverb::new(*room_size, *damping, *mix))
            }
            EffectConfig::LowPass { cutoff, resonance } => {
                Box::new(Filter::lowpass(*cutoff, *resonance))
            }
            EffectConfig::HighPass { cutoff, resonance } => {
                Box::new(Filter::highpass(*cutoff, *resonance))
            }
            EffectConfig::BandPass { cutoff, resonance } => {
                Box::new(Filter::bandpass(*cutoff, *resonance))
            }
        }
    }
}

/// Builder functions for effect configs (used in effects![] macro)
pub fn distortion(drive: f32, mix: f32) -> EffectConfig {
    EffectConfig::Distortion { drive, mix, output: 1.0 }
}

pub fn distortion_full(drive: f32, mix: f32, output: f32) -> EffectConfig {
    EffectConfig::Distortion { drive, mix, output }
}

pub fn delay(time: f32, feedback: f32, mix: f32) -> EffectConfig {
    EffectConfig::Delay { time, feedback, mix }
}

pub fn chorus(rate: f32, depth: f32, mix: f32) -> EffectConfig {
    EffectConfig::Chorus { rate, depth, mix }
}

pub fn flanger(rate: f32, depth: f32, feedback: f32, mix: f32) -> EffectConfig {
    EffectConfig::Flanger { rate, depth, feedback, mix }
}

pub fn reverb(room_size: f32, damping: f32, mix: f32) -> EffectConfig {
    EffectConfig::Reverb { room_size, damping, mix }
}

pub fn lowpass(cutoff: f32, resonance: f32) -> EffectConfig {
    EffectConfig::LowPass { cutoff, resonance }
}

pub fn highpass(cutoff: f32, resonance: f32) -> EffectConfig {
    EffectConfig::HighPass { cutoff, resonance }
}

pub fn bandpass(cutoff: f32, resonance: f32) -> EffectConfig {
    EffectConfig::BandPass { cutoff, resonance }
}

// =============================================================================
// EFFECT TRAIT
// =============================================================================

/// Trait for all DSP effects
pub trait Effect: Send + Sync {
    /// Process a single sample, return the processed sample
    fn process(&mut self, sample: f32) -> f32;
    /// Reset the effect state
    fn reset(&mut self);
}

// =============================================================================
// DISTORTION
// =============================================================================

/// Hard clipping distortion with harmonic enhancement
#[derive(Clone)]
pub struct Distortion {
    pub drive: f32,      // 1.0 = clean, 10.0+ = heavy distortion
    pub mix: f32,        // 0.0 = dry, 1.0 = wet
    pub output: f32,     // Output gain compensation
    sample_count: usize,
    debug_printed: bool,
}

impl Default for Distortion {
    fn default() -> Self {
        Self {
            drive: 4.0,
            mix: 1.0,
            output: 1.0,
            sample_count: 0,
            debug_printed: false,
        }
    }
}

impl Distortion {
    pub fn new(drive: f32) -> Self {
        Self {
            drive,
            ..Default::default()
        }
    }
}

impl Effect for Distortion {
    fn process(&mut self, sample: f32) -> f32 {
        // Normalize input: boost quiet signals to usable level
        // Typical synth output is around 0.001-0.01, we want to work with ~1.0
        let normalized = sample * 200.0;  // Boost to working level

        // Apply drive (with drive=100, even small signals will clip hard)
        let driven = normalized * self.drive;

        // Hard clipping at ±1.0
        let clipped = driven.clamp(-1.0, 1.0);

        // Asymmetric clipping for tube-like even harmonics
        let asymmetric = if clipped > 0.0 {
            clipped
        } else {
            clipped * 0.85
        };

        // Scale back to audible level
        let wet = asymmetric * 0.1 * self.output;

        sample * (1.0 - self.mix) + wet * self.mix
    }

    fn reset(&mut self) {
        self.sample_count = 0;
        self.debug_printed = false;
    }
}

// =============================================================================
// DELAY
// =============================================================================

/// Simple delay effect
#[derive(Clone)]
pub struct Delay {
    buffer: Vec<f32>,
    write_pos: usize,
    pub delay_time: f32,  // In seconds
    pub feedback: f32,    // 0.0 to 1.0
    pub mix: f32,         // 0.0 = dry, 1.0 = wet
}

impl Delay {
    pub fn new(delay_time: f32, feedback: f32, mix: f32) -> Self {
        let buffer_size = (delay_time * SAMPLE_RATE) as usize + 1;
        Self {
            buffer: vec![0.0; buffer_size.max(1)],
            write_pos: 0,
            delay_time,
            feedback: feedback.clamp(0.0, 0.95),
            mix,
        }
    }
}

impl Default for Delay {
    fn default() -> Self {
        Self::new(0.25, 0.4, 0.5)
    }
}

impl Effect for Delay {
    fn process(&mut self, sample: f32) -> f32 {
        let delay_samples = (self.delay_time * SAMPLE_RATE) as usize;
        let read_pos = (self.write_pos + self.buffer.len() - delay_samples) % self.buffer.len();

        let delayed = self.buffer[read_pos];
        self.buffer[self.write_pos] = sample + delayed * self.feedback;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();

        sample * (1.0 - self.mix) + delayed * self.mix
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }
}

// =============================================================================
// CHORUS
// =============================================================================

/// Chorus effect using modulated delay
#[derive(Clone)]
pub struct Chorus {
    buffer: Vec<f32>,
    write_pos: usize,
    phase: f32,
    pub rate: f32,        // LFO rate in Hz
    pub depth: f32,       // Modulation depth in samples
    pub mix: f32,
}

impl Chorus {
    pub fn new(rate: f32, depth: f32, mix: f32) -> Self {
        let buffer_size = (0.05 * SAMPLE_RATE) as usize; // 50ms max delay
        Self {
            buffer: vec![0.0; buffer_size],
            write_pos: 0,
            phase: 0.0,
            rate,
            depth: depth * SAMPLE_RATE * 0.003, // Convert to samples (max 3ms)
            mix,
        }
    }
}

impl Default for Chorus {
    fn default() -> Self {
        Self::new(1.5, 0.5, 0.5)
    }
}

impl Effect for Chorus {
    fn process(&mut self, sample: f32) -> f32 {
        // Store sample
        self.buffer[self.write_pos] = sample;

        // LFO modulation
        let lfo = (self.phase * 2.0 * std::f32::consts::PI).sin();
        let delay_samples = 10.0 + self.depth * (1.0 + lfo) * 0.5;

        // Linear interpolation for fractional delay
        let read_pos_f = self.write_pos as f32 - delay_samples;
        let read_pos_f = if read_pos_f < 0.0 { read_pos_f + self.buffer.len() as f32 } else { read_pos_f };
        let read_pos_i = read_pos_f as usize % self.buffer.len();
        let frac = read_pos_f.fract();
        let next_pos = (read_pos_i + 1) % self.buffer.len();

        let delayed = self.buffer[read_pos_i] * (1.0 - frac) + self.buffer[next_pos] * frac;

        // Advance LFO
        self.phase += self.rate / SAMPLE_RATE;
        if self.phase >= 1.0 { self.phase -= 1.0; }

        self.write_pos = (self.write_pos + 1) % self.buffer.len();

        sample * (1.0 - self.mix) + delayed * self.mix
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
        self.phase = 0.0;
    }
}

// =============================================================================
// FLANGER
// =============================================================================

/// Flanger effect - short modulated delay with feedback
#[derive(Clone)]
pub struct Flanger {
    buffer: Vec<f32>,
    write_pos: usize,
    phase: f32,
    pub rate: f32,        // LFO rate in Hz
    pub depth: f32,       // 0.0 to 1.0
    pub feedback: f32,    // -1.0 to 1.0
    pub mix: f32,
}

impl Flanger {
    pub fn new(rate: f32, depth: f32, feedback: f32, mix: f32) -> Self {
        let buffer_size = (0.02 * SAMPLE_RATE) as usize; // 20ms max
        Self {
            buffer: vec![0.0; buffer_size],
            write_pos: 0,
            phase: 0.0,
            rate,
            depth,
            feedback: feedback.clamp(-0.95, 0.95),
            mix,
        }
    }
}

impl Default for Flanger {
    fn default() -> Self {
        Self::new(0.5, 0.7, 0.7, 0.5)
    }
}

impl Effect for Flanger {
    fn process(&mut self, sample: f32) -> f32 {
        let lfo = (self.phase * 2.0 * std::f32::consts::PI).sin();
        let delay_samples = 1.0 + (self.depth * (self.buffer.len() - 2) as f32 * 0.5) * (1.0 + lfo);

        // Linear interpolation
        let read_pos_f = self.write_pos as f32 - delay_samples;
        let read_pos_f = if read_pos_f < 0.0 { read_pos_f + self.buffer.len() as f32 } else { read_pos_f };
        let read_pos_i = read_pos_f as usize % self.buffer.len();
        let frac = read_pos_f.fract();
        let next_pos = (read_pos_i + 1) % self.buffer.len();

        let delayed = self.buffer[read_pos_i] * (1.0 - frac) + self.buffer[next_pos] * frac;

        self.buffer[self.write_pos] = sample + delayed * self.feedback;

        self.phase += self.rate / SAMPLE_RATE;
        if self.phase >= 1.0 { self.phase -= 1.0; }

        self.write_pos = (self.write_pos + 1) % self.buffer.len();

        sample * (1.0 - self.mix) + delayed * self.mix
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
        self.phase = 0.0;
    }
}

// =============================================================================
// REVERB (Simple Schroeder)
// =============================================================================

/// Simple reverb using comb and allpass filters
#[derive(Clone)]
pub struct Reverb {
    // Parallel comb filters
    comb_buffers: [Vec<f32>; 4],
    comb_positions: [usize; 4],
    comb_feedback: f32,
    // Series allpass filters
    allpass_buffers: [Vec<f32>; 2],
    allpass_positions: [usize; 2],
    pub mix: f32,
    pub room_size: f32,
    pub damping: f32,
    lpf_state: f32,
}

impl Reverb {
    pub fn new(room_size: f32, damping: f32, mix: f32) -> Self {
        // Prime number delays for comb filters (in samples at 44.1kHz)
        let comb_delays: [usize; 4] = [1557, 1617, 1491, 1422];
        let allpass_delays: [usize; 2] = [225, 341];

        let scale = room_size.clamp(0.5, 2.0);

        Self {
            comb_buffers: [
                vec![0.0; (comb_delays[0] as f32 * scale) as usize],
                vec![0.0; (comb_delays[1] as f32 * scale) as usize],
                vec![0.0; (comb_delays[2] as f32 * scale) as usize],
                vec![0.0; (comb_delays[3] as f32 * scale) as usize],
            ],
            comb_positions: [0; 4],
            comb_feedback: 0.84,
            allpass_buffers: [
                vec![0.0; (allpass_delays[0] as f32 * scale) as usize],
                vec![0.0; (allpass_delays[1] as f32 * scale) as usize],
            ],
            allpass_positions: [0; 2],
            mix,
            room_size: scale,
            damping: damping.clamp(0.0, 1.0),
            lpf_state: 0.0,
        }
    }
}

impl Default for Reverb {
    fn default() -> Self {
        Self::new(1.0, 0.5, 0.3)
    }
}

impl Effect for Reverb {
    fn process(&mut self, sample: f32) -> f32 {
        let input = sample;

        // Process parallel comb filters
        let mut comb_out = 0.0;
        for i in 0..4 {
            let buf = &mut self.comb_buffers[i];
            let pos = self.comb_positions[i];

            let delayed = buf[pos];
            // Low-pass filter in feedback path for damping
            self.lpf_state = delayed * (1.0 - self.damping) + self.lpf_state * self.damping;
            buf[pos] = input + self.lpf_state * self.comb_feedback;

            self.comb_positions[i] = (pos + 1) % buf.len();
            comb_out += delayed;
        }
        comb_out *= 0.25;

        // Process series allpass filters
        let mut allpass_out = comb_out;
        for i in 0..2 {
            let buf = &mut self.allpass_buffers[i];
            let pos = self.allpass_positions[i];

            let delayed = buf[pos];
            let temp = allpass_out + delayed * 0.5;
            buf[pos] = allpass_out;
            allpass_out = delayed - temp * 0.5;

            self.allpass_positions[i] = (pos + 1) % buf.len();
        }

        sample * (1.0 - self.mix) + allpass_out * self.mix
    }

    fn reset(&mut self) {
        for buf in &mut self.comb_buffers {
            buf.fill(0.0);
        }
        for buf in &mut self.allpass_buffers {
            buf.fill(0.0);
        }
        self.comb_positions = [0; 4];
        self.allpass_positions = [0; 2];
        self.lpf_state = 0.0;
    }
}

// =============================================================================
// FILTERS (Resonant State Variable Filter)
// =============================================================================

/// Filter type
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FilterType {
    LowPass,
    HighPass,
    BandPass,
}

/// State Variable Filter - resonant multimode filter
#[derive(Clone)]
pub struct Filter {
    pub filter_type: FilterType,
    pub cutoff: f32,      // Cutoff frequency in Hz
    pub resonance: f32,   // 0.0 to 1.0 (self-oscillation near 1.0)
    // State variables
    low: f32,
    band: f32,
    high: f32,
}

impl Filter {
    pub fn new(filter_type: FilterType, cutoff: f32, resonance: f32) -> Self {
        Self {
            filter_type,
            cutoff: cutoff.clamp(20.0, 20000.0),
            resonance: resonance.clamp(0.0, 0.99),
            low: 0.0,
            band: 0.0,
            high: 0.0,
        }
    }

    pub fn lowpass(cutoff: f32, resonance: f32) -> Self {
        Self::new(FilterType::LowPass, cutoff, resonance)
    }

    pub fn highpass(cutoff: f32, resonance: f32) -> Self {
        Self::new(FilterType::HighPass, cutoff, resonance)
    }

    pub fn bandpass(cutoff: f32, resonance: f32) -> Self {
        Self::new(FilterType::BandPass, cutoff, resonance)
    }
}

impl Default for Filter {
    fn default() -> Self {
        Self::lowpass(5000.0, 0.0)
    }
}

impl Effect for Filter {
    fn process(&mut self, sample: f32) -> f32 {
        // Calculate coefficients
        let f = 2.0 * (std::f32::consts::PI * self.cutoff / SAMPLE_RATE).sin();
        let q = 1.0 - self.resonance;

        // Two-pole state variable filter
        self.high = sample - self.low - q * self.band;
        self.band += f * self.high;
        self.low += f * self.band;

        match self.filter_type {
            FilterType::LowPass => self.low,
            FilterType::HighPass => self.high,
            FilterType::BandPass => self.band,
        }
    }

    fn reset(&mut self) {
        self.low = 0.0;
        self.band = 0.0;
        self.high = 0.0;
    }
}

// =============================================================================
// EFFECT CHAIN
// =============================================================================

/// A chain of effects to apply in series
pub struct EffectChain {
    effects: Vec<Box<dyn Effect>>,
}

impl EffectChain {
    pub fn new() -> Self {
        Self { effects: Vec::new() }
    }

    /// Add an effect to the chain
    pub fn add<E: Effect + 'static>(&mut self, effect: E) -> &mut Self {
        self.effects.push(Box::new(effect));
        self
    }

    /// Process a sample through all effects
    pub fn process(&mut self, sample: f32) -> f32 {
        let mut out = sample;
        for effect in &mut self.effects {
            out = effect.process(out);
        }
        out
    }

    /// Reset all effects
    pub fn reset(&mut self) {
        for effect in &mut self.effects {
            effect.reset();
        }
    }

    /// Check if chain is empty
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }
}

impl Default for EffectChain {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// CHANNEL EFFECTS PROCESSOR
// =============================================================================

/// Manages per-channel effect chains
pub struct ChannelEffects {
    channels: [EffectChain; 16],
}

impl ChannelEffects {
    pub fn new() -> Self {
        Self {
            channels: std::array::from_fn(|_| EffectChain::new()),
        }
    }

    /// Add an effect to a specific channel
    pub fn add_effect<E: Effect + 'static>(&mut self, channel: u8, effect: E) {
        if (channel as usize) < 16 {
            self.channels[channel as usize].add(effect);
        }
    }

    /// Process a sample for a specific channel
    pub fn process(&mut self, channel: u8, sample: f32) -> f32 {
        if (channel as usize) < 16 {
            self.channels[channel as usize].process(sample)
        } else {
            sample
        }
    }

    /// Reset all channel effects
    pub fn reset(&mut self) {
        for chain in &mut self.channels {
            chain.reset();
        }
    }

    /// Check if a channel has effects
    pub fn has_effects(&self, channel: u8) -> bool {
        if (channel as usize) < 16 {
            !self.channels[channel as usize].is_empty()
        } else {
            false
        }
    }
}

impl Default for ChannelEffects {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distortion() {
        let mut dist = Distortion::new(5.0);
        let out = dist.process(0.5);
        assert!(out.abs() < 1.0); // Should be clipped
    }

    #[test]
    fn test_delay() {
        let mut delay = Delay::new(0.01, 0.5, 1.0);
        // First sample should be silent (only delayed)
        let out1 = delay.process(1.0);
        assert!(out1.abs() < 0.01);
        // After delay time (0.01s = 441 samples at 44100Hz), should hear the signal
        // Need to process exactly delay_samples more times to reach the echo
        let delay_samples = (0.01 * super::SAMPLE_RATE) as usize;
        let mut found_echo = false;
        for _ in 0..delay_samples {
            let out = delay.process(0.0);
            if out.abs() > 0.5 {
                found_echo = true;
                break;
            }
        }
        assert!(found_echo, "Should hear delayed signal after delay time");
    }

    #[test]
    fn test_filter() {
        let mut lpf = Filter::lowpass(1000.0, 0.0);
        // Process a DC signal
        for _ in 0..100 {
            lpf.process(1.0);
        }
        let out = lpf.process(1.0);
        assert!(out > 0.9); // DC should pass through lowpass
    }

    #[test]
    fn test_effect_chain() {
        let mut chain = EffectChain::new();
        chain.add(Distortion::new(2.0));
        chain.add(Filter::lowpass(5000.0, 0.0));

        let out = chain.process(0.5);
        assert!(out.abs() < 1.0);
    }
}
