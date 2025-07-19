import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def prikazi_pet_z_masko(image_path, mask_precuneus, mask_cerebellum, slice_idx=None, view='axial'):
    image = nib.load(image_path + ".img").get_fdata()
    image = np.squeeze(image)

    # Preuredi maske v (x, y, z)
    mask_precuneus = np.transpose(mask_precuneus, (1, 0, 2))
    mask_cerebellum = np.transpose(mask_cerebellum, (1, 0, 2))

    # Spremeni pogled (orientacijo prereza)
    if view == 'sagittal':
        image = np.transpose(image, (2, 1, 0))
        mask_precuneus = np.transpose(mask_precuneus, (2, 1, 0))
        mask_cerebellum = np.transpose(mask_cerebellum, (2, 1, 0))
    elif view == 'coronal':
        image = np.transpose(image, (0, 2, 1))
        mask_precuneus = np.transpose(mask_precuneus, (0, 2, 1))
        mask_cerebellum = np.transpose(mask_cerebellum, (0, 2, 1))
    # če view == 'axial', ostane kot je

    if slice_idx is None:
        slice_idx = image.shape[2] // 2  # sredinski prerez


    plt.figure(figsize=(16, 4))

    # (a) Celotna slika
    plt.subplot(1, 4, 1)
    im1 = plt.imshow(image[:, :, slice_idx], cmap='hot')
    plt.title("Originalna slika možganov")
    plt.axis('off')
    plt.colorbar(im1, fraction=0.046, pad=0.04)

    # (b.i) Precuneus maska
    plt.subplot(1, 4, 2)
    plt.imshow(image[:, :, slice_idx], cmap='gray')
    plt.imshow(mask_precuneus[:, :, slice_idx], cmap='Reds', alpha=0.4)
    plt.title("Precuneus regija")
    plt.axis('off')

    # (b.ii) Cerebellum maska
    plt.subplot(1, 4, 3)
    plt.imshow(image[:, :, slice_idx], cmap='gray')
    plt.imshow(mask_cerebellum[:, :, slice_idx], cmap='Blues', alpha=0.4)
    plt.title("Cerebellum regija")
    plt.axis('off')

    # (a+b skupaj)
    plt.subplot(1, 4, 4)
    combined_mask = np.clip(mask_precuneus + mask_cerebellum, 0, 1)
    plt.imshow(image[:, :, slice_idx], cmap='gray')
    plt.imshow(combined_mask[:, :, slice_idx], cmap='spring', alpha=0.4)
    plt.title("Obe regiji skupaj")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Primer: naloži maske (če še niso naložene)
mask_data = np.load("Masks/mask_resized_reshaped.npy", allow_pickle=True).item()
mask_precuneus = mask_data["mask_precuneus"]
mask_cerebellum = mask_data["mask_cerebellum"]



# Prikaz ene slike (npr. CN primer)
prikazi_pet_z_masko(
    "FDG PET images/AD-patients/w002_S_5018S",
    mask_precuneus,
    mask_cerebellum,
    slice_idx=40,
    view='coronal'  # ali 'sagittal', 'axial'
)

def primerjaj_ad_cn(ad_path, cn_path, mask_precuneus, mask_cerebellum, slice_idx=None, save_path=None):
    def load_and_prepare(path):
        img = np.squeeze(nib.load(path + ".img").get_fdata())
        return img

    # Preuredi maske (vnaprej)
    m_prec = np.transpose(mask_precuneus, (1, 0, 2))
    m_cereb = np.transpose(mask_cerebellum, (1, 0, 2))
    combined_mask = np.clip(m_prec + m_cereb, 0, 1)

    ad_img = load_and_prepare(ad_path)
    cn_img = load_and_prepare(cn_path)

    if slice_idx is None:
        slice_idx = ad_img.shape[2] // 2  # sredinski prerez

    plt.figure(figsize=(10, 5))

    # AD bolnik
    plt.subplot(1, 2, 1)
    plt.imshow(cn_img[:, :, slice_idx], cmap='gray')
    plt.imshow(combined_mask[:, :, slice_idx], cmap='spring', alpha=0.4)
    plt.title("CN - kontrolni preiskovanec")
    plt.axis('off')

    # CN bolnik
    plt.subplot(1, 2, 2)
    plt.imshow(ad_img[:, :, slice_idx], cmap='gray')
    plt.imshow(combined_mask[:, :, slice_idx], cmap='spring', alpha=0.4)
    plt.title("AD - bolnik z Alzheimerjevo boleznijo")
    plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

# Primer primerjave CN in AD slike
primerjaj_ad_cn(
    "FDG PET images/AD-patients/w002_S_5018S",
    "FDG PET images/CN-controls/w002_S_0413S",
    mask_precuneus,
    mask_cerebellum,
    slice_idx=39,
    save_path="primerjava_ad_vs_cn.png"
)

def primerjaj_vec_parov(ad_paths, cn_paths, mask_precuneus, mask_cerebellum, slice_idx=39, output_folder="primerjave/"):
    import os
    os.makedirs(output_folder, exist_ok=True)

    for i, (ad_path, cn_path) in enumerate(zip(ad_paths, cn_paths)):
        save_name = os.path.basename(ad_path) + "_vs_" + os.path.basename(cn_path) + ".png"
        save_path = os.path.join(output_folder, f"primerjava_{i+1}_{save_name}")
        print(f"Shranjujem primerjavo {i+1}: {save_path}")
        primerjaj_ad_cn(ad_path, cn_path, mask_precuneus, mask_cerebellum, slice_idx=slice_idx, save_path=save_path)

# Seznami treh primerov (lahko zamenjaš s katerimikoli)
ad_primeri = [
    "FDG PET images/AD-patients/w002_S_5018S",
    "FDG PET images/AD-patients/w019_S_5019S",
    "FDG PET images/AD-patients/w130_S_5059S",
]

cn_primeri = [
    "FDG PET images/CN-controls/w002_S_0413S",
    "FDG PET images/CN-controls/w002_S_4213S",
    "FDG PET images/CN-controls/w019_S_4835S",  # <- veljaven primer
]

# Klic funkcije
primerjaj_vec_parov(ad_primeri, cn_primeri, mask_precuneus, mask_cerebellum, slice_idx=39)
