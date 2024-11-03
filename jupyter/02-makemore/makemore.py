import torch
import matplotlib.pyplot as plt

words = open("./jupyter/02-makemore/names.txt", "r").read().splitlines()

N = torch.zeros((27, 27), dtype=torch.int32)
chars = sorted(list(set("".join(words))))
stoi = { x:(i+1) for i, x in enumerate(chars)}
stoi["."] = 0
itos = { i:s for s,i in stoi.items() }

count = {}
for w in words[:]:
    c = ["."] + list(w) + ["."]
    for i in range(len(c))[:-1]:
        N[stoi[c[i]], stoi[c[i+1]]] += 1

plt.figure(figsize=(16, 16))
plt.imshow(N, cmap="Blues")
for i in range(27):
    for j in range(27):
        c = itos[i] + itos[j]
        plt.text(j, i, c, ha="center", va="bottom", color="gray")
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
plt.axis("off");

# Probabilty distribution for starting character.
p = N[0].float()
p = p / p.sum()
p
