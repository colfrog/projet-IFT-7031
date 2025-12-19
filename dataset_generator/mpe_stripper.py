"""
MPE to Plain MIDI Converter
Strips MPE expression data from MIDI files to create plain MIDI suitable for model training

Usage:
    python mpe_stripper.py --input generated_midi/guitar --output plain_midi/guitar
    python mpe_stripper.py --help
"""

import mido
from mido import Message, MidiFile, MidiTrack
from pathlib import Path
import shutil
import argparse


class MPEStripper:
    """
    Strip MPE-specific expression data from MIDI files
    
    Keeps: notes, timing, velocity, tempo
    Removes: aftertouch (pressure), pitch bends (optional), CC74 (timbre)
    """
    
    def __init__(self, keep_pitch_bends=False):
        """
        Initialize MPE stripper
        
        Args:
            keep_pitch_bends: If True, keep pitch bend messages (like Basic Pitch)
                            If False, strip all expression including pitch bends
        """
        # Message types to keep (basic note information)
        self.keep_types = {
            'note_on',
            'note_off',
            'set_tempo',
            'time_signature',
            'key_signature',
            'track_name',
            'end_of_track',
            'program_change'
        }
        
        if keep_pitch_bends:
            self.keep_types.add('pitchwheel')
        
        # MPE-specific messages to remove
        self.remove_types = {
            'aftertouch',      # Channel pressure (MPE expression)
            'polytouch',       # Polyphonic aftertouch (MPE expression)
        }
        
        # Control changes to remove (MPE timbre control)
        self.remove_cc = {74}  # CC74 = timbre/brightness in MPE
    
    def strip_mpe_from_file(self, mpe_file, output_file=None, 
                           keep_velocity=True, quantize=False):
        """
        Strip MPE expression data from a MIDI file
        
        Args:
            mpe_file: Path to MPE MIDI file
            output_file: Path for output (if None, auto-generate name)
            keep_velocity: If True, preserve note velocities
            quantize: If True, quantize timing to 16th notes
        
        Returns:
            Path to created plain MIDI file
        """
        mpe_file = Path(mpe_file)
        
        if output_file is None:
            output_file = mpe_file.parent / f"{mpe_file.stem}_plain.mid"
        
        mid = mido.MidiFile(mpe_file)
        plain = MidiFile(ticks_per_beat=mid.ticks_per_beat)
        
        for track in mid.tracks:
            new_track = MidiTrack()
            plain.tracks.append(new_track)
            
            accumulated_time = 0
            
            for msg in track:
                accumulated_time += msg.time
                
                # Skip MPE-specific expression messages
                if msg.type in self.remove_types:
                    continue
                
                # Skip MPE timbre control (CC74)
                if msg.type == 'control_change' and hasattr(msg, 'control'):
                    if msg.control in self.remove_cc:
                        continue
                
                # Keep selected message types
                if msg.type in self.keep_types or msg.type == 'control_change':
                    if msg.type in ['note_on', 'note_off']:
                        # Handle velocity
                        if not keep_velocity and hasattr(msg, 'velocity'):
                            new_msg = msg.copy(
                                velocity=80 if msg.type == 'note_on' else 0,
                                time=accumulated_time
                            )
                        else:
                            new_msg = msg.copy(time=accumulated_time)
                        
                        # Handle quantization
                        if quantize:
                            quantize_unit = mid.ticks_per_beat // 4
                            new_msg.time = round(new_msg.time / quantize_unit) * quantize_unit
                        
                        new_track.append(new_msg)
                    else:
                        new_msg = msg.copy(time=accumulated_time)
                        new_track.append(new_msg)
                    
                    accumulated_time = 0
        
        plain.save(output_file)
        return output_file
    
    def process_directory(self, input_dir, output_dir=None, 
                         keep_velocity=True, quantize=False):
        """
        Process all MIDI files in a directory
        
        Args:
            input_dir: Directory containing MPE MIDI files
            output_dir: Directory for plain MIDI (if None, create 'plain' subdirectory)
            keep_velocity: Preserve note velocities
            quantize: Quantize timing to 16th notes
            
        Returns:
            Number of files processed
        """
        input_dir = Path(input_dir)
        
        if output_dir is None:
            output_dir = input_dir / 'plain'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        midi_files = list(input_dir.glob('*.mid')) + list(input_dir.glob('*.midi'))
        
        if not midi_files:
            print(f"⚠️  No MIDI files found in {input_dir}")
            return 0
        
        print(f"Processing {len(midi_files)} MIDI files...")
        print(f"Input:  {input_dir}")
        print(f"Output: {output_dir}")
        print()
        
        processed = 0
        for i, midi_file in enumerate(midi_files):
            output_file = output_dir / f"{midi_file.stem}_plain.mid"
            
            try:
                self.strip_mpe_from_file(
                    midi_file, 
                    output_file,
                    keep_velocity=keep_velocity,
                    quantize=quantize
                )
                processed += 1
                
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(midi_files)} files...")
            
            except Exception as e:
                print(f"  ⚠️  Error processing {midi_file.name}: {e}")
        
        print(f"\n✅ Processed {processed} files")
        print(f"   Output: {output_dir}")
        
        return processed
    
    def create_training_dataset(self, midi_dir, audio_dir, output_dir):
        """
        Organize MIDI and audio files into training dataset structure
        
        Input structure:
            midi_dir/
                guitar/
                    file_001.mid
                    file_002.mid
            audio_dir/
                guitar/
                    file_001.wav
                    file_002.wav
        
        Output structure:
            output_dir/
                guitar/
                    sample_0000/
                        audio.wav
                        mpe.mid
                        plain.mid
                    sample_0001/
                        ...
        
        Args:
            midi_dir: Directory containing MIDI files organized by instrument
            audio_dir: Directory containing audio files organized by instrument
            output_dir: Output directory for organized dataset
            
        Returns:
            Total number of samples organized
        """
        midi_dir = Path(midi_dir)
        audio_dir = Path(audio_dir)
        output_dir = Path(output_dir)
        
        print("Creating training dataset structure...")
        print("=" * 60)
        
        total_samples = 0
        
        for instrument_dir in midi_dir.iterdir():
            if not instrument_dir.is_dir():
                continue
            
            instrument = instrument_dir.name
            print(f"\nProcessing {instrument}...")
            
            # Check for existing samples
            instrument_output_dir = output_dir / instrument
            existing_samples = []
            if instrument_output_dir.exists():
                existing_samples = list(instrument_output_dir.glob('sample_*'))
            
            start_count = len(existing_samples)
            if start_count > 0:
                print(f"  Found {start_count} existing samples, continuing from {start_count}...")
            
            # Get MIDI and audio files
            midi_files = sorted(instrument_dir.glob('*.mid'))
            audio_instrument_dir = audio_dir / instrument
            
            if not audio_instrument_dir.exists():
                print(f"  ⚠️  Audio directory not found: {audio_instrument_dir}")
                continue
            
            count = start_count
            added = 0
            
            for midi_file in midi_files:
                # Find corresponding audio file
                audio_file = audio_instrument_dir / f"{midi_file.stem}.wav"
                
                if not audio_file.exists():
                    # Try different audio extensions
                    audio_file = audio_instrument_dir / f"{midi_file.stem}.mp3"
                    if not audio_file.exists():
                        audio_file = audio_instrument_dir / f"{midi_file.stem}.flac"
                        if not audio_file.exists():
                            print(f"  ⚠️  No audio for {midi_file.name}, skipping...")
                            continue
                
                # Create sample directory
                sample_dir = output_dir / instrument / f"sample_{count:04d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy audio
                shutil.copy(audio_file, sample_dir / f"audio{audio_file.suffix}")
                
                # Copy MPE MIDI
                shutil.copy(midi_file, sample_dir / "mpe.mid")
                
                # Create plain MIDI
                self.strip_mpe_from_file(
                    midi_file,
                    sample_dir / "plain.mid",
                    keep_velocity=True,
                    quantize=False
                )
                
                count += 1
                added += 1
                
                if added % 100 == 0:
                    print(f"  Organized {added} new samples...")
            
            print(f"  ✅ Added {added} {instrument} samples (total: {count})")
            total_samples += count
        
        print()
        print("=" * 60)
        print(f"✅ Training dataset created: {total_samples} total samples")
        print(f"   Location: {output_dir.absolute()}")
        
        return total_samples


def compare_midi_files(mpe_file, plain_file):
    """
    Compare MPE and plain MIDI files to show what was stripped
    
    Args:
        mpe_file: Path to MPE MIDI file
        plain_file: Path to plain MIDI file
    """
    mpe = mido.MidiFile(mpe_file)
    plain = mido.MidiFile(plain_file)
    
    print("MPE MIDI Analysis:")
    print("=" * 50)
    
    mpe_msg_types = {}
    for track in mpe.tracks:
        for msg in track:
            mpe_msg_types[msg.type] = mpe_msg_types.get(msg.type, 0) + 1
    
    print("Message types in MPE:")
    for msg_type, count in sorted(mpe_msg_types.items()):
        print(f"  {msg_type:20s}: {count:5d}")
    
    print("\nPlain MIDI Analysis:")
    print("=" * 50)
    
    plain_msg_types = {}
    for track in plain.tracks:
        for msg in track:
            plain_msg_types[msg.type] = plain_msg_types.get(msg.type, 0) + 1
    
    print("Message types in Plain:")
    for msg_type, count in sorted(plain_msg_types.items()):
        print(f"  {msg_type:20s}: {count:5d}")
    
    print("\nRemoved (MPE expression):")
    print("=" * 50)
    removed_count = 0
    for msg_type in mpe_msg_types:
        if msg_type not in plain_msg_types:
            print(f"  {msg_type:20s}: {mpe_msg_types[msg_type]:5d} messages removed")
            removed_count += mpe_msg_types[msg_type]
    
    print(f"\nTotal messages removed: {removed_count}")


def main():
    """Main entry point with command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description='Strip MPE expression data from MIDI files'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Strip command
    strip_parser = subparsers.add_parser('strip', help='Strip MPE from files')
    strip_parser.add_argument('input', type=str, help='Input MIDI file or directory')
    strip_parser.add_argument('--output', '-o', type=str, help='Output file or directory')
    strip_parser.add_argument('--keep-pitch-bends', action='store_true',
                             help='Keep pitch bend messages (like Basic Pitch)')
    strip_parser.add_argument('--normalize-velocity', action='store_true',
                             help='Normalize all velocities to 80')
    strip_parser.add_argument('--quantize', action='store_true',
                             help='Quantize timing to 16th notes')
    
    # Organize command
    organize_parser = subparsers.add_parser('organize', help='Organize into training dataset')
    organize_parser.add_argument('--midi', type=str, required=True,
                                help='Directory containing MIDI files')
    organize_parser.add_argument('--audio', type=str, required=True,
                                help='Directory containing audio files')
    organize_parser.add_argument('--output', '-o', type=str, required=True,
                                help='Output directory for training dataset')
    organize_parser.add_argument('--keep-pitch-bends', action='store_true',
                                help='Keep pitch bend messages')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare MPE and plain MIDI')
    compare_parser.add_argument('mpe_file', type=str, help='MPE MIDI file')
    compare_parser.add_argument('plain_file', type=str, help='Plain MIDI file')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == 'strip':
        stripper = MPEStripper(keep_pitch_bends=args.keep_pitch_bends)
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Single file
            output = args.output if args.output else None
            stripper.strip_mpe_from_file(
                input_path,
                output,
                keep_velocity=not args.normalize_velocity,
                quantize=args.quantize
            )
            print(f"✅ Stripped: {output or input_path.parent / (input_path.stem + '_plain.mid')}")
        
        elif input_path.is_dir():
            # Directory
            stripper.process_directory(
                input_path,
                args.output,
                keep_velocity=not args.normalize_velocity,
                quantize=args.quantize
            )
        else:
            print(f"❌ Error: {input_path} not found")
    
    elif args.command == 'organize':
        stripper = MPEStripper(keep_pitch_bends=args.keep_pitch_bends)
        total = stripper.create_training_dataset(
            midi_dir=args.midi,
            audio_dir=args.audio,
            output_dir=args.output
        )
        print(f"\n✅ Organized {total} samples")
    
    elif args.command == 'compare':
        compare_midi_files(args.mpe_file, args.plain_file)


if __name__ == "__main__":
    main()