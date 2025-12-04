from cityscapes_dataset import CityscapesFineDataset

ROOT = r"C:\Users\dowen\Desktop\csci 490dpl final project"

ds = CityscapesFineDataset(ROOT, split="val")
print("Dataset length (val):", len(ds))

img, gt, path = ds[0]
print("Sample path:", path)
print("Image tensor shape:", img.shape)
print("GT tensor shape:", gt.shape)
print("GT unique labels (first few):", gt.unique()[:10])
