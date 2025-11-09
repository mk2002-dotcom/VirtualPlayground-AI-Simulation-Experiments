# DNA RNA protein
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# DNA配列（例：短い遺伝子）
length = 90  # DNAの長さ（コドン単位なら3の倍数がおすすめ）
DNA = "".join(random.choice("ATGC") for _ in range(length))

# 転写表（DNA → mRNA）
transcription = {"A":"U", "T":"A", "G":"C", "C":"G"}

# コドン表（mRNAの3塩基 → アミノ酸）
codon_table = {
    # START & STOP
    "AUG": "Met", "UAA": "STOP", "UAG": "STOP", "UGA": "STOP",
    # Phenylalanine, Leucine
    "UUU":"Phe", "UUC":"Phe", "UUA":"Leu", "UUG":"Leu",
    "CUU":"Leu", "CUC":"Leu", "CUA":"Leu", "CUG":"Leu",
    # Isoleucine, Valine, Serine
    "AUU":"Ile", "AUC":"Ile", "AUA":"Ile",
    "GUU":"Val", "GUC":"Val", "GUA":"Val", "GUG":"Val",
    "UCU":"Ser", "UCC":"Ser", "UCA":"Ser", "UCG":"Ser",
    "AGU":"Ser", "AGC":"Ser",
    # Proline, Threonine, Alanine
    "CCU":"Pro", "CCC":"Pro", "CCA":"Pro", "CCG":"Pro",
    "ACU":"Thr", "ACC":"Thr", "ACA":"Thr", "ACG":"Thr",
    "GCU":"Ala", "GCC":"Ala", "GCA":"Ala", "GCG":"Ala",
    # Tyrosine, Histidine, Glutamine, Asparagine
    "UAU":"Tyr", "UAC":"Tyr", "CAU":"His", "CAC":"His",
    "CAA":"Gln", "CAG":"Gln", "AAU":"Asn", "AAC":"Asn",
    # Aspartic acid, Glutamic acid
    "GAU":"Asp", "GAC":"Asp", "GAA":"Glu", "GAG":"Glu",
    # Cysteine, Tryptophan, Arginine, Glycine
    "UGU":"Cys", "UGC":"Cys", "UGG":"Trp",
    "CGU":"Arg", "CGC":"Arg", "CGA":"Arg", "CGG":"Arg",
    "AGA":"Arg", "AGG":"Arg",
    "GGU":"Gly", "GGC":"Gly", "GGA":"Gly", "GGG":"Gly",
    # Lysine
    "AAA":"Lys", "AAG":"Lys"
}

# 転写: DNA → RNA
RNA = "".join(transcription[b] for b in DNA)

# グラフ初期設定
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(0, len(DNA))
ax.set_ylim(0, 3)
ax.axis("off")

dna_text = ax.text(0, 2.3, "", fontsize=12, color="blue")
rna_text = ax.text(0, 1.3, "", fontsize=12, color="orange")
protein_text = ax.text(0, 0.3, "", fontsize=12, color="green")

def animate(i):
    n = i + 1
    dna_text.set_text("DNA: " + DNA[:n])
    rna_text.set_text("mRNA: " + RNA[:n])
    # 翻訳（3文字ずつ）
    if n >= 3:
        codons = [RNA[j:j+3] for j in range(0, n, 3) if len(RNA[j:j+3])==3]
        amino_acids = []
        for c in codons:
            aa = codon_table.get(c, "")
            if aa == "STOP":
                break
            if aa:
                amino_acids.append(aa)
        protein_text.set_text("Protein: " + "-".join(amino_acids))
    return dna_text, rna_text, protein_text

ani = animation.FuncAnimation(fig, animate, frames=len(DNA), interval=300, blit=True)
plt.show()

