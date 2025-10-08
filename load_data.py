import synapseclient 
import synapseutils 
import os
import imageio.v2 as imageio
from pathlib import Path
import numpy as np
import nibabel as nib


from settings import SYS_TOKEN, BASE


parent_id = "syn3193805"  # projeto raiz
pastas_desejadas = {
    "averaged-testing-images",
    "averaged-training-images",
    "averaged-training-labels",
}

data_path = os.path.join(BASE, "data", "synapse")
os.makedirs(data_path, exist_ok=True)

def load_from_synapse():
    syn = synapseclient.Synapse() 
    syn.login(authToken=SYS_TOKEN) 
    for ch in syn.getChildren(parent_id):
        if ch["type"] == "org.sagebionetworks.repo.model.Folder" and ch["name"] in pastas_desejadas:
            destino = os.path.join(data_path, ch["name"])
            os.makedirs(destino, exist_ok=True)
            print(f"Baixando: {ch['name']} ({ch['id']}) → {destino}")
            synapseutils.syncFromSynapse(
                syn,
                ch["id"],
                path=destino,              
                ifcollision="overwrite.local",
                followLink=True
            )

def decompress_gz():
    import gzip
    import shutil
    for root, dirs, files in os.walk(data_path):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            for file in os.listdir(dir_path):
                if file.endswith(".gz"):
                    gz_path = os.path.join(dir_path, file)
                    out_path = os.path.join(dir_path, file[:-3])  # remove .gz
                    print(f"Decompressing {gz_path} to {out_path}")
                    with gzip.open(gz_path, 'rb') as f_in:
                        with open(out_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(gz_path)  


def to_uint8(img2d, p_low=1, p_high=99):
    """Normaliza para uint8 usando percentis (evita dividir por zero e melhora contraste)."""
    x = np.asarray(img2d, dtype=np.float32)
    # Lida com constantes / NaN
    if not np.isfinite(x).any():
        return np.zeros_like(x, dtype=np.uint8)
    lo, hi = np.percentile(x[np.isfinite(x)], [p_low, p_high])
    if hi <= lo:
        lo, hi = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(x, dtype=np.uint8)
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    return (x * 255).astype(np.uint8)

def _move_channel_last(arr):
    """
    Se existir um eixo com tamanho 3 (RGB) que não é o último, move-o para o fim.
    Não mexe se já estiver adequado.
    """
    if arr.ndim >= 3:
        for ax in range(arr.ndim - 1):
            if arr.shape[ax] == 3:
                axes = [i for i in range(arr.ndim) if i != ax] + [ax]
                return np.transpose(arr, axes)
    return arr

def nii_to_jpgs(input_path, output_dir, rgb=False, ext="jpg"):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = nib.load(str(input_path)).get_fdata()
    data = np.squeeze(np.asarray(data))
    data = _move_channel_last(data)  # tenta garantir canal por último

    # Casos:
    # 2D: (H, W)
    # 3D: (H, W, D) -> D slices, 1 canal
    # 4D: (H, W, D, C) ou (H, W, C, D) -> tentamos canal por último
    if data.ndim == 2:
        # um único slice, canal único
        ch_dir = output_dir / "channel_0"
        ch_dir.mkdir(parents=True, exist_ok=True)
        img8 = to_uint8(data)
        imageio.imwrite(str(ch_dir / f"channel_0_slice_0.{ext}"), img8)
        return

    if data.ndim == 3:
        H, W, D = data.shape
        ch_dir = output_dir / "channel_0"
        ch_dir.mkdir(parents=True, exist_ok=True)
        for z in range(D):
            slice2d = data[..., z]
            img8 = to_uint8(slice2d)
            if rgb:
                img8 = np.stack([img8, img8, img8], axis=-1)  # H x W x 3
            imageio.imwrite(str(ch_dir / f"channel_0_slice_{z}.{ext}"), img8)
        return

    if data.ndim == 4:
        H, W, A, B = data.shape  # tentaremos D=A, C=B
        D, C = A, B

        # Se acharmos que o canal está na penúltima dimensão (ex.: (H, W, 3, D)), invertemos:
        if C > 4 and D <= 4:
            # provavelmente (H, W, C, D) com C pequeno; traz C para o fim
            data = np.moveaxis(data, -2, -1)  # agora (H, W, D, C)
            H, W, D, C = data.shape

        # Agora assumimos (H, W, D, C)
        for c in range(C):
            ch_dir = output_dir / f"channel_{c}"
            ch_dir.mkdir(parents=True, exist_ok=True)
            for z in range(D):
                slice2d = np.squeeze(data[..., z, c])
                if slice2d.ndim != 2:
                    # segurança extra
                    print(f"[WARN] slice {z} canal {c} shape {slice2d.shape} não é 2D; pulando.")
                    continue
                img8 = to_uint8(slice2d)
                if rgb and C == 3:
                    # Se tivermos exatamente 3 canais e rgb=True, você pode preferir salvar 1 imagem RGB por slice
                    # Mas mantendo sua lógica de "por canal", só empilhamos se pediu rgb explicitamente
                    img8 = np.stack([img8, img8, img8], axis=-1)
                imageio.imwrite(str(ch_dir / f"channel_{c}_slice_{z}.{ext}"), img8)
        return

    raise ValueError(f"Dimensão NIfTI não suportada: shape={data.shape}")


for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.lower().endswith(".nii") or file.lower().endswith(".nii.gz"):
            input_path = Path(root) / file
            relative_path = os.path.relpath(root, data_path)
            out_dir = Path(data_path) / "jpgs" / relative_path / file.replace(".nii.gz", "").replace(".nii", "")
            print(f"Converting {input_path} to JPGs in {out_dir}")
            nii_to_jpgs(input_path, out_dir, rgb=False, ext="jpg")


