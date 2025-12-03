import os
import json
import py_midicsv as pm

data_info = []
for instrument in os.listdir("training_data"):
    for sample in os.listdir(f"training_data/{instrument}"):
        midi_csv = ''.join(pm.midi_to_csv(f"training_data/{instrument}/{sample}/plain.mid"))
        mpe_csv = ''.join(pm.midi_to_csv(f"training_data/{instrument}/{sample}/mpe.mid"))
        data_info.append({
            "audio": f"/home/laurent/scratch/training_data/{instrument}/{sample}/audio.wav",
            "instruction": f"Convert this MIDI to MPE\n\n{midi_csv}",
            "output": mpe_csv
        })

with open("train_data.json", "w") as f:
    json.dump(data_info, f)

print("JSON file written")