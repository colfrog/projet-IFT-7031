"""
MPE MIDI Generator for Expressive Instruments
Generates realistic expressive MIDI files with pitch bends, aftertouch, and timbre control

Usage:
    python mpe_midi_generator.py --output generated_midi --samples 1000
"""

import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
import numpy as np
from pathlib import Path
import argparse


class MPEMIDIGenerator:
    """Generate MPE MIDI files with realistic expressive techniques"""
    
    def __init__(self, output_dir='generated_midi', tempo_bpm=120):
        """
        Initialize MPE MIDI generator
        
        Args:
            output_dir: Directory to save generated MIDI files
            tempo_bpm: Tempo in beats per minute
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # MIDI timing
        self.ticks_per_beat = 480
        self.tempo = int(60_000_000 / tempo_bpm)  # Microseconds per beat
        
    def create_base_track(self):
        """Create a MIDI track with basic setup"""
        mid = MidiFile(ticks_per_beat=self.ticks_per_beat)
        track = MidiTrack()
        mid.tracks.append(track)
        
        track.append(MetaMessage('set_tempo', tempo=self.tempo))
        track.append(MetaMessage('track_name', name='MPE Track'))
        
        return mid, track
    
    def ticks(self, seconds):
        """Convert seconds to MIDI ticks based on current tempo"""
        beats = seconds / (self.tempo / 1_000_000)
        return int(beats * self.ticks_per_beat)
    
    @staticmethod
    def clamp_pitch(value):
        """Clamp pitch bend value to valid MIDI range"""
        return max(-8192, min(8191, int(value)))
    
    # ========== GUITAR TECHNIQUES ==========
    
    def generate_guitar_bend(self, note=64, bend_semitones=2.0, duration=2.0, channel=1):
        """
        Generate a guitar bend (bend up and release)
        
        Args:
            note: MIDI note number (0-127)
            bend_semitones: Amount to bend in semitones
            duration: Total duration in seconds
            channel: MIDI channel (1-16)
        """
        mid, track = self.create_base_track()
        
        track.append(Message('note_on', note=note, velocity=90, channel=channel, time=0))
        
        # Bend up phase (first half of duration)
        bend_duration = duration / 2
        steps = 50
        for i in range(steps):
            progress = i / steps
            curve = np.power(progress, 0.7)  # Exponential curve for natural feel
            bend_value = self.clamp_pitch(curve * bend_semitones * 4096 / 12)
            track.append(Message('pitchwheel', pitch=bend_value, channel=channel, 
                               time=self.ticks(bend_duration / steps)))
        
        # Hold at top
        track.append(Message('pitchwheel', pitch=self.clamp_pitch(bend_semitones * 4096 / 12), 
                           channel=channel, time=self.ticks(0.3)))
        
        # Release phase (bend back down)
        for i in range(steps):
            progress = i / steps
            curve = 1 - np.power(progress, 0.7)
            bend_value = self.clamp_pitch(curve * bend_semitones * 4096 / 12)
            track.append(Message('pitchwheel', pitch=bend_value, channel=channel,
                               time=self.ticks(bend_duration / steps)))
        
        track.append(Message('note_off', note=note, velocity=0, channel=channel, 
                           time=self.ticks(0.1)))
        
        return mid
    
    def generate_guitar_slide(self, start_note=60, end_note=67, duration=1.5, channel=1):
        """Generate a slide between two notes"""
        mid, track = self.create_base_track()
        
        track.append(Message('note_on', note=start_note, velocity=85, channel=channel, time=0))
        
        semitone_diff = end_note - start_note
        steps = 60
        for i in range(steps):
            progress = i / steps
            # S-curve for natural slide feel
            curve = (np.tanh((progress - 0.5) * 4) + 1) / 2
            bend_value = self.clamp_pitch(curve * semitone_diff * 4096 / 12)
            track.append(Message('pitchwheel', pitch=bend_value, channel=channel,
                               time=self.ticks(duration / steps)))
        
        # Hold at destination
        track.append(Message('pitchwheel', pitch=self.clamp_pitch(semitone_diff * 4096 / 12),
                           channel=channel, time=self.ticks(0.3)))
        
        track.append(Message('note_off', note=start_note, velocity=0, channel=channel, time=0))
        
        return mid
    
    def generate_guitar_vibrato(self, note=67, duration=3.0, vibrato_rate=5.0, 
                                vibrato_depth=0.5, channel=1):
        """Generate vibrato (periodic pitch oscillation)"""
        mid, track = self.create_base_track()
        
        track.append(Message('note_on', note=note, velocity=85, channel=channel, time=0))
        
        steps = int(duration * 100)  # 100 samples per second
        for i in range(steps):
            t = i / 100
            vibrato = np.sin(2 * np.pi * vibrato_rate * t) * vibrato_depth
            bend_value = self.clamp_pitch(vibrato * 4096 / 12)
            
            # Add pressure variation synchronized with vibrato
            pressure = int(64 + 20 * np.sin(2 * np.pi * vibrato_rate * t))
            
            track.append(Message('pitchwheel', pitch=bend_value, channel=channel,
                               time=self.ticks(1.0 / 100)))
            track.append(Message('aftertouch', value=pressure, channel=channel, time=0))
        
        track.append(Message('note_off', note=note, velocity=0, channel=channel, time=0))
        
        return mid
    
    def generate_guitar_hammer_on(self, notes=None, channel=1):
        """Generate hammer-on/pull-off sequence"""
        if notes is None:
            notes = [60, 62, 64]
            
        mid, track = self.create_base_track()
        
        # First note with full attack
        track.append(Message('note_on', note=notes[0], velocity=90, channel=channel, time=0))
        track.append(Message('note_off', note=notes[0], velocity=0, channel=channel,
                           time=self.ticks(0.3)))
        
        # Subsequent notes with softer attack (hammer-on effect)
        for i, note in enumerate(notes[1:], 1):
            prev_note = notes[i - 1]
            semitone_diff = note - prev_note
            
            # Quick slide to simulate hammer-on
            steps = 10
            for j in range(steps):
                progress = j / steps
                bend = self.clamp_pitch(progress * semitone_diff * 4096 / 12)
                track.append(Message('pitchwheel', pitch=bend, channel=channel,
                                   time=self.ticks(0.02)))
            
            track.append(Message('note_on', note=note, velocity=60, channel=channel, time=0))
            track.append(Message('pitchwheel', pitch=0, channel=channel, time=0))
            track.append(Message('note_off', note=note, velocity=0, channel=channel,
                               time=self.ticks(0.3)))
        
        return mid
    
    # ========== VIOLIN TECHNIQUES ==========
    
    def generate_violin_vibrato(self, note=69, duration=4.0, vibrato_rate=6.5,
                                vibrato_depth=0.3, channel=1):
        """Generate violin vibrato (faster and narrower than guitar)"""
        mid, track = self.create_base_track()
        
        track.append(Message('note_on', note=note, velocity=80, channel=channel, time=0))
        
        steps = int(duration * 100)
        for i in range(steps):
            t = i / 100
            # Gradual vibrato depth increase (natural violin technique)
            depth_envelope = min(1.0, t / 0.5)
            current_depth = vibrato_depth * depth_envelope
            
            vibrato = np.sin(2 * np.pi * vibrato_rate * t) * current_depth
            bend_value = self.clamp_pitch(vibrato * 4096 / 12)
            
            # Pressure variation synchronized with vibrato
            pressure = int(70 + 15 * np.sin(2 * np.pi * vibrato_rate * t))
            
            track.append(Message('pitchwheel', pitch=bend_value, channel=channel,
                               time=self.ticks(1.0 / 100)))
            track.append(Message('aftertouch', value=pressure, channel=channel, time=0))
        
        track.append(Message('note_off', note=note, velocity=0, channel=channel, time=0))
        
        return mid
    
    def generate_violin_portamento(self, start_note=64, end_note=71, duration=1.2, channel=1):
        """Generate smooth violin portamento (glissando)"""
        mid, track = self.create_base_track()
        
        track.append(Message('note_on', note=start_note, velocity=75, channel=channel, time=0))
        
        semitone_diff = end_note - start_note
        steps = 80
        for i in range(steps):
            progress = i / steps
            # Smooth S-curve
            curve = (1 - np.cos(progress * np.pi)) / 2
            
            # Main pitch movement with subtle vibrato
            pitch_shift = curve * semitone_diff
            t = i / 100
            vibrato = 0.1 * np.sin(2 * np.pi * 6 * t)
            
            total_shift = pitch_shift + vibrato
            bend_value = self.clamp_pitch(total_shift * 4096 / 12)
            
            # Pressure increases during portamento
            pressure = int(60 + 30 * curve)
            
            track.append(Message('pitchwheel', pitch=bend_value, channel=channel,
                               time=self.ticks(duration / steps)))
            track.append(Message('aftertouch', value=pressure, channel=channel, time=0))
        
        track.append(Message('note_off', note=start_note, velocity=0, channel=channel, time=0))
        
        return mid
    
    def generate_violin_bow_pressure(self, note=67, duration=3.0, channel=1):
        """Generate bow pressure variation (dynamics)"""
        mid, track = self.create_base_track()
        
        track.append(Message('note_on', note=note, velocity=70, channel=channel, time=0))
        
        steps = int(duration * 50)
        for i in range(steps):
            progress = i / steps
            # Bell curve for natural crescendo-decrescendo
            pressure_curve = np.exp(-((progress - 0.5) ** 2) / 0.1)
            pressure = int(40 + 80 * pressure_curve)
            
            # Timbre changes with pressure (MPE standard CC74)
            timbre = int(40 + 60 * pressure_curve)
            
            track.append(Message('aftertouch', value=pressure, channel=channel,
                               time=self.ticks(duration / steps)))
            track.append(Message('control_change', control=74, value=timbre,
                               channel=channel, time=0))
        
        track.append(Message('note_off', note=note, velocity=0, channel=channel, time=0))
        
        return mid
    
    # ========== BATCH GENERATION ==========
    
    def generate_guitar_dataset(self, num_samples=1000, note_range=(55, 75)):
        """
        Generate diverse guitar training samples
        
        Args:
            num_samples: Total number of samples to generate
            note_range: Tuple of (min_note, max_note) for randomization
        """
        print(f"Generating {num_samples} guitar MIDI files...")
        
        guitar_dir = self.output_dir / 'guitar'
        guitar_dir.mkdir(exist_ok=True)
        
        techniques = ['bend', 'slide', 'vibrato', 'hammer_on']
        samples_per_technique = num_samples // len(techniques)
        
        count = 0
        
        # Bends
        for _ in range(samples_per_technique):
            note = np.random.randint(*note_range)
            bend = np.random.uniform(0.5, 2.5)
            duration = np.random.uniform(1.5, 3.0)
            
            mid = self.generate_guitar_bend(note, bend, duration)
            mid.save(guitar_dir / f'guitar_bend_{count:04d}.mid')
            count += 1
        
        # Slides
        for _ in range(samples_per_technique):
            start = np.random.randint(note_range[0], note_range[1] - 10)
            end = start + np.random.randint(3, 10)
            duration = np.random.uniform(1.0, 2.0)
            
            mid = self.generate_guitar_slide(start, end, duration)
            mid.save(guitar_dir / f'guitar_slide_{count:04d}.mid')
            count += 1
        
        # Vibrato
        for _ in range(samples_per_technique):
            note = np.random.randint(*note_range)
            rate = np.random.uniform(4.0, 7.0)
            depth = np.random.uniform(0.3, 0.8)
            duration = np.random.uniform(2.0, 4.0)
            
            mid = self.generate_guitar_vibrato(note, duration, rate, depth)
            mid.save(guitar_dir / f'guitar_vibrato_{count:04d}.mid')
            count += 1
        
        # Hammer-ons
        for _ in range(samples_per_technique):
            start_note = np.random.randint(note_range[0], note_range[1] - 6)
            notes = [start_note, start_note + 2, start_note + 4]
            
            mid = self.generate_guitar_hammer_on(notes)
            mid.save(guitar_dir / f'guitar_hammer_{count:04d}.mid')
            count += 1
        
        print(f"✅ Generated {count} guitar MIDI files in {guitar_dir}")
        return count
    
    def generate_violin_dataset(self, num_samples=1000, note_range=(62, 84)):
        """
        Generate diverse violin training samples
        
        Args:
            num_samples: Total number of samples to generate
            note_range: Tuple of (min_note, max_note) for randomization
        """
        print(f"Generating {num_samples} violin MIDI files...")
        
        violin_dir = self.output_dir / 'violin'
        violin_dir.mkdir(exist_ok=True)
        
        techniques = ['vibrato', 'portamento', 'bow_pressure']
        samples_per_technique = num_samples // len(techniques)
        
        count = 0
        
        # Vibrato (generate more samples for this important technique)
        for _ in range(samples_per_technique * 2):
            note = np.random.randint(*note_range)
            rate = np.random.uniform(5.5, 7.5)
            depth = np.random.uniform(0.2, 0.5)
            duration = np.random.uniform(2.5, 5.0)
            
            mid = self.generate_violin_vibrato(note, duration, rate, depth)
            mid.save(violin_dir / f'violin_vibrato_{count:04d}.mid')
            count += 1
        
        # Portamento
        for _ in range(samples_per_technique):
            start = np.random.randint(note_range[0], note_range[1] - 7)
            end = start + np.random.randint(-7, 10)
            end = np.clip(end, *note_range)
            duration = np.random.uniform(0.8, 1.8)
            
            mid = self.generate_violin_portamento(start, end, duration)
            mid.save(violin_dir / f'violin_portamento_{count:04d}.mid')
            count += 1
        
        # Bow pressure
        for _ in range(samples_per_technique):
            note = np.random.randint(*note_range)
            duration = np.random.uniform(2.0, 4.0)
            
            mid = self.generate_violin_bow_pressure(note, duration)
            mid.save(violin_dir / f'violin_bow_{count:04d}.mid')
            count += 1
        
        print(f"✅ Generated {count} violin MIDI files in {violin_dir}")
        return count


def main():
    """Main entry point with command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description='Generate MPE MIDI files with realistic expressive techniques'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='generated_midi',
        help='Output directory for generated MIDI files (default: generated_midi)'
    )
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=1000,
        help='Number of samples to generate per instrument (default: 1000)'
    )
    parser.add_argument(
        '--tempo', '-t',
        type=int,
        default=120,
        help='Tempo in BPM (default: 120)'
    )
    parser.add_argument(
        '--instruments', '-i',
        nargs='+',
        choices=['guitar', 'violin', 'all'],
        default=['all'],
        help='Instruments to generate (default: all)'
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = MPEMIDIGenerator(output_dir=args.output, tempo_bpm=args.tempo)
    
    print("=" * 60)
    print("MPE MIDI GENERATOR")
    print("=" * 60)
    print(f"Output directory: {Path(args.output).absolute()}")
    print(f"Samples per instrument: {args.samples}")
    print(f"Tempo: {args.tempo} BPM")
    print()
    
    total_generated = 0
    instruments = args.instruments if 'all' not in args.instruments else ['guitar', 'violin']
    
    # Generate datasets
    if 'guitar' in instruments:
        total_generated += generator.generate_guitar_dataset(num_samples=args.samples)
        print()
    
    if 'violin' in instruments:
        total_generated += generator.generate_violin_dataset(num_samples=args.samples)
        print()
    
    print("=" * 60)
    print("✅ Generation complete!")
    print(f"Total files generated: {total_generated}")
    print(f"Location: {Path(args.output).absolute()}")


if __name__ == "__main__":
    main()