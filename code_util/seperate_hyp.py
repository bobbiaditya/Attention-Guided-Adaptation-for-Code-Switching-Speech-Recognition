import os
import subprocess
import sys

# Check if the correct number of command-line arguments is provided
if len(sys.argv) != 3:
    print("Usage: python separate_sentences.py [hyp_file_path] [ref_file_path]")
    sys.exit(1)

# Get the input file paths from command-line arguments
hyp_file_path = sys.argv[1]
ref_file_path = sys.argv[2]

# Create the new_eval folder if it doesn't exist
output_folder = "new_eval"
os.makedirs(output_folder, exist_ok=True)

# Read the reference file
with open(ref_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Initialize empty lists for code-switched, English, and Mandarin sentences
cs_sentences = []
en_sentences = []
man_sentences = []
cs_tags = []
en_tags = []
man_tags = []

# Process each line in the reference file
for line in lines:
    line = line.strip()
    sentence, tag = line.split('\t')
    
    if all('\u4e00' <= c <= '\u9fff' for c in sentence.split()):
        man_sentences.append(line)
        man_tags.append(tag)
    elif any('\u4e00' <= c <= '\u9fff' for c in sentence.split()):
        cs_sentences.append(line)
        cs_tags.append(tag)
    else:
        en_sentences.append(line)
        en_tags.append(tag)

# Write code-switched sentences to file
with open(os.path.join(output_folder, 'ref.trn.cs'), 'w', encoding='utf-8') as file:
    file.write('\n'.join(cs_sentences))

# Write English sentences to file
with open(os.path.join(output_folder, 'ref.trn.en'), 'w', encoding='utf-8') as file:
    file.write('\n'.join(en_sentences))

# Write Mandarin sentences to file
with open(os.path.join(output_folder, 'ref.trn.man'), 'w', encoding='utf-8') as file:
    file.write('\n'.join(man_sentences))


# Read the hypothesis file
with open(hyp_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

ref_cs_sentences = []
ref_en_sentences = []
ref_man_sentences = []

# Process each line in the hypothesis file
for line in lines:
    line = line.strip()
    if len(line.split('\t'))==1:
        tag = line.split('\t')[0]
        # print(tag)
    else:
        sentence, tag = line.split('\t')
    
    if tag in man_tags:
        ref_man_sentences.append(line)
    elif tag in cs_tags:
        ref_cs_sentences.append(line)
    else:
        ref_en_sentences.append(line)

# Write code-switched sentences to file
with open(os.path.join(output_folder, 'hyp.trn.cs'), 'w', encoding='utf-8') as file:
    file.write('\n'.join(ref_cs_sentences))

# Write English sentences to file
with open(os.path.join(output_folder, 'hyp.trn.en'), 'w', encoding='utf-8') as file:
    file.write('\n'.join(ref_en_sentences))

# Write Mandarin sentences to file
with open(os.path.join(output_folder, 'hyp.trn.man'), 'w', encoding='utf-8') as file:
    file.write('\n'.join(ref_man_sentences))

# print(os.path.join(output_folder, 'hyp.trn.man'))
# Run the SCLITE commands
command1 = "/home/espnet/tools/sctk/bin/sclite -e utf-8 -c NOASCII -r new_eval/ref.trn.man trn -h new_eval/hyp.trn.man trn -i rm -o all stdout > new_eval/result.man.txt"
command2 = "/home/espnet/tools/sctk/bin/sclite -e utf-8 -c NOASCII -r new_eval/ref.trn.en trn -h new_eval/hyp.trn.en trn -i rm -o all stdout > new_eval/result.en.txt"
command3 = "/home/espnet/tools/sctk/bin/sclite -e utf-8 -c NOASCII -r new_eval/ref.trn.cs trn -h new_eval/hyp.trn.cs trn -i rm -o all stdout > new_eval/result.cs.txt"

subprocess.run(command1, shell=True)
subprocess.run(command2, shell=True)
subprocess.run(command3, shell=True)