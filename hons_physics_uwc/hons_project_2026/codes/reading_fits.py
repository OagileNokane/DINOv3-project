from astropy.io import fits
import torchvision, torch, os
import numpy as np
import pathlib
from transformers import AutoImageProcessor, AutoModel
from datetime import datetime
import pandas as pd
from PIL import Image

start_time = datetime.now()

main_dir = "/home/oagile/hons_physics_uwc/hons_project_2026"  # Directory of all files
output_dir = os.path.join(main_dir, "dinov3_features_extracted_testing")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading dinov3
model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, device_map="auto")
model.eval()

total = 165416  # Last fits file number
batch_size = 50  # Process 50 images simultaneously for massive speedup
chunk_size = 5000  # Save a checkpoint chunk to disk every 5000 successfully processed files

batch_images = []
batch_ids = []
current_chunk_features = []
current_chunk_ids = []
chunk_counter = 0
processed_count = 0

def save_chunk(features_list, ids_list, chunk_idx):
  if not features_list:
    return
  feature_matrix = np.concatenate(features_list, axis=0)

  # Create a DataFrame for this chunk
  chunk_df = pd.DataFrame(list(zip(ids_list, feature_matrix)), columns=["CatId", "features"])

  chunk_file = os.path.join(output_dir, f"features_chunk_{chunk_idx}.csv")
  chunk_df.to_csv(chunk_file, index=False)
  print(f"\n[Checkpoint] Saved chunk {chunk_idx} to: {chunk_file}\n")


print("Starting batched feature extraction...")


# Looping to load every fits file
for i in range(5, total+1): 
    # Path for the fits file
    file_path = pathlib.Path(f"/home/oagile/hons_physics_uwc/hons_project_2026/data/RGZ_fits_cropped/images_full/{i}.fits")
    # Checking if the fits file exists or not
    if file_path.is_file():
        try:
          hdul = fits.open(file_path)
          img_data = hdul[0].data  # Image data 

          img = np.nan_to_num(img_data)
          v_min, v_max = np.percentile(img, (1, 99))
          img = np.clip(img, v_min, v_max)
          if v_max > v_min:
            img = (img - v_min) / (v_max - v_min)
          else:
            img = np.zeros_like(img)  # Dinov3 requires a 3-channel image
      

          # Converting to 3-channel
          img_3ch = np.stack([img, img, img], axis=-1)
          img_rgb = (img_3ch * 255).astype(np.uint8)
          img_pil = Image.fromarray(img_rgb)

          batch_images.append(img_pil)
          batch_ids.append(i)
          processed_count += 1

          # Processing the images when batch number is met
          if len(batch_images) == batch_size:
            inputs = processor(images=batch_images, return_tensors="pt").to(
            model.device
          )
            
            with torch.inference_mode():
              outputs = model(**inputs)
              cls_tokens = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            current_chunk_features.append(cls_tokens)
            current_chunk_ids.extend(batch_ids)

            if len(current_chunk_ids) >= chunk_size:
              save_chunk(current_chunk_features, current_chunk_ids, chunk_counter)
              chunk_counter += 1

              # Resetting chunk containers
              current_chunk_features = []
              current_chunk_ids = []

            print(f"Processed fits file ID {i} | Total successfully processed:{processed_count} and {99566-processed_count} are left")

            # Resetting batch containers
            batch_images = []
            batch_ids = []

        except Exception as e:
          print(f"Error processing file {i}: {e}")


if batch_images:
  inputs = processor(images=batch_images, return_tensors="pt").to(model.device)
  with torch.inference_mode():
    outputs = model(**inputs)
    cls_tokens = outputs.last_hidden_state[:, 0, :].cpu().numpy()

  current_chunk_features.append(cls_tokens)
  current_chunk_ids.extend(batch_ids)

# Save the final remaining chunk
if current_chunk_ids:
  save_chunk(current_chunk_features, current_chunk_ids, chunk_counter)
  print(f"Processed fits file ID {i} | Total successfully processed:{processed_count} and {99566-processed_count} are left")


print(f"It took {datetime.now()-start_time} to do feature extraction!!!")

