from miditok import REMI, TokenizerConfig, TokSequence
from pathlib import Path
import mido
import os

DATA_PATH = Path("training_data")
MIDI_PATHS = DATA_PATH.glob("**/*.mid")
SAMPLE_PATHS = DATA_PATH.glob("*/*")
REMI_PATH = "remi.json"


def explode_mpe_to_tracks(input_midi_path, output_midi_path):
    """
    Splits an MPE MIDI file (usually Type 0 or 1 merged) into
    separate tracks for each Channel (1-16).
    This forces the Tokenizer to treat them as separate 'Voices'.
    """
    mid = mido.MidiFile(input_midi_path)
    new_mid = mido.MidiFile()
    new_mid.ticks_per_beat = mid.ticks_per_beat

    # Create 17 empty tracks (0=Global, 1-16=Channels)
    channels = [mido.MidiTrack() for _ in range(17)]

    # Sort every message into its corresponding track based on channel
    for track in mid.tracks:
        for msg in track:
            if hasattr(msg, 'channel'):
                # Channels are 0-15 in mido, mapped to tracks 1-16
                channels[msg.channel + 1].append(msg)
            else:
                # Meta messages (Tempo, Time Sig) go to Track 0 (Global)
                channels[0].append(msg)

    # Add non-empty tracks to the new file
    for track in channels:
        if len(track) > 0:
            new_mid.tracks.append(track)

    new_mid.save(output_midi_path)
    print(f"Exploded MPE file to {len(new_mid.tracks)} tracks.")

class RemiCompactor():
    def __init__(self):
        if os.path.exists(REMI_PATH):
            self.remi = REMI(params=Path(REMI_PATH))
        else:
            remi_config = TokenizerConfig(
                num_velocities=16,
                use_pitch_bends=True,
                pitch_bend_range=(-8192, 8191, 1024), # We need a lot of values for pitch bends
                use_control_changes=True,
                use_programs=True,
                one_token_stream_for_programs=False,
                use_time_signatures=True,
                ac_polyphony_track=True
            )
            self.remi = REMI(remi_config)
            self.remi.train(vocab_size=30000, files_paths=MIDI_PATHS + MUSICNET_MIDI_PATHS)
            self.remi.save(REMI_PATH)

    def midi_to_str(self, path, mpe=False):
        if mpe:
            explode_mpe_to_tracks(path, "/tmp/exploded.mid")
            path = "/tmp/exploded.mid"
        tokens = self.remi.encode(path)
        if isinstance(tokens, list):
            tokens = tokens[0]

        self.remi.complete_sequence(tokens)
        return " ".join(tokens.tokens)

    def str_to_midi(self, output, output_path="output/mpe.mid"):
        # 1. Split string into individual token strings
        token_strings = output.strip().split(' ')

        # 2. Convert String Tokens -> Integer IDs
        # We need to look up the ID for each token string in the vocab
        token_ids = []
        for t in token_strings:
            if t in self.remi.vocab:
                token_ids.append(self.remi.vocab[t])
            else:
                print(f"Warning: Skipping unknown token '{t}'")

        # 3. Create a TokSequence
        # MidiTok v3 requires a TokSequence object for decoding
        seq = TokSequence(ids=token_ids)

        # 4. Decode (IDs -> MIDI)
        # If BPE was used, this automatically handles the BPE decoding
        midi = self.remi.decode([seq])

        # 5. Save
        midi.dump_midi(output_path)
        print(f"Successfully saved MIDI to {output_path}")

sample = 'training_data/guitar/sample_0819'
remi = RemiCompactor()
mpe = remi.midi_to_str(f"{sample}/mpe.mid", mpe=True)
midi = remi.midi_to_str(f"{sample}/plain.mid")

print(mpe)