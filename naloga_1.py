import os
import numpy as np
import nibabel as nib
from scipy.ndimage import label
from scipy import ndimage


# Poti do datotek
ad_dir = "FDG PET images/AD-patients"
cn_dir = "FDG PET images/CN-controls"
mask_path = "Masks/mask_resized_reshaped.npy"

# Nalaganje maske
mask_data = np.load(mask_path, allow_pickle=True).item()

print("Ključi v mask_data:", mask_data.keys())

mask_precuneus = mask_data["mask_precuneus"]
mask_cerebellum = mask_data["mask_cerebellum"]



# Pomožna funkcija za nalaganje Analyze .hdr/.img formata
def load_analyze_image(filepath_no_ext):
    img = nib.load(filepath_no_ext + ".img")
    return img.get_fdata()


def get_brain_mask(image, threshold=0.1):
    return image > threshold


def adjust_mask(mask, target_shape):
    """Preuredi osi maske, če je potrebno, da se ujema s PET sliko"""
    if mask.shape != target_shape:
        return np.transpose(mask, (1, 0, 2))  # iz (95, 79, 78) → (79, 95, 78)
    return mask


# Pridobi poti do vseh .img datotek
def get_image_paths(folder):
    return [os.path.join(folder, f)[:-4] for f in os.listdir(folder) if f.endswith('.img')]

# Izračun SUV vrednosti
def compute_suv_metrics(image, mask):
    values = image[mask > 0]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    suv_max = np.max(values)
    suv_mean = np.mean(values)

    # SUVpeak = max average v 1 cm³ – privzamemo 3x3x3 voxlov kot približek
    max_peak = 0
    labeled, _ = ndimage.label(mask)
    for z in range(1, image.shape[2] - 1):
        for y in range(1, image.shape[1] - 1):
            for x in range(1, image.shape[0] - 1):
                if mask[x, y, z] > 0:
                    cube = image[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]
                    avg = np.mean(cube)
                    max_peak = max(max_peak, avg)
    return suv_max, suv_mean, max_peak


# Pridobi poti do vseh slik
ad_paths = get_image_paths(ad_dir)
cn_paths = get_image_paths(cn_dir)

# Združi poti in ustvari oznake (1 = AD, 0 = CN)
all_samples = [(path, 1) for path in ad_paths] + [(path, 0) for path in cn_paths]

# Po želji: sortiraš, da imaš konsistenten vrstni red
all_samples.sort()

# Primer izpisa prvih nekaj
for path, label in all_samples[:5]:
    print("Slika:", path, "| Oznaka:", label)

import pandas as pd

results = []

for path, label in all_samples:
    image = load_analyze_image(path)
    image = np.squeeze(image)  # odstrani kanalno dimenzijo → (79, 95, 78)

    brain_mask = get_brain_mask(image)

    # Prilagodi maske dimenzijam slike
    mask_precuneus_adj = adjust_mask(mask_precuneus, image.shape)
    mask_cerebellum_adj = adjust_mask(mask_cerebellum, image.shape)

    # Izračun SUV vrednosti
    suv_brain = compute_suv_metrics(image, brain_mask)
    suv_precuneus = compute_suv_metrics(image, mask_precuneus_adj)
    suv_cerebellum = compute_suv_metrics(image, mask_cerebellum_adj)

    results.append({
        "filename": os.path.basename(path),
        "label": label,
        "SUVmax_brain": suv_brain[0],
        "SUVmean_brain": suv_brain[1],
        "SUVpeak_brain": suv_brain[2],
        "SUVmax_precuneus": suv_precuneus[0],
        "SUVmean_precuneus": suv_precuneus[1],
        "SUVpeak_precuneus": suv_precuneus[2],
        "SUVmax_cerebellum": suv_cerebellum[0],
        "SUVmean_cerebellum": suv_cerebellum[1],
        "SUVpeak_cerebellum": suv_cerebellum[2],
    })

# Shrani kot DataFrame in izvozi v CSV
df = pd.DataFrame(results)
df.to_csv("rezultati_suv.csv", index=False)

print("✅ Rezultati shranjeni v datoteko: rezultati_suv.csv")


