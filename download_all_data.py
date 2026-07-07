import os
import urllib.request

def download_complete_abide_dataset():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Download the official phenotypic master mapping file
    pheno_url = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv"
    pheno_path = os.path.join(data_dir, "phenotypic.csv")
    
    print("⏳ Syncing with the master phenotypic database...")
    if not os.path.exists(pheno_path):
        urllib.request.urlretrieve(pheno_url, pheno_path)
    
    # 2. Extract every single valid subject ID and its scanning site name
    # The ABIDE S3 bucket organizes files by prefixing the scanning site name (e.g., Pitt, KKI, Olin, NYU)
    subjects_to_download = []
    with open(pheno_path, "r") as f:
        lines = f.readlines()
        headers = lines[0].split(",")
        
        # Locate indices for SITE_ID and SUB_ID columns
        site_idx = headers.index("SITE_ID")
        sub_idx = headers.index("SUB_ID")
        
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) > max(site_idx, sub_idx):
                site_name = parts[site_idx].strip()
                sub_id = parts[sub_idx].strip()
                if sub_id.isdigit():
                    subjects_to_download.append((site_name, sub_id))

    total_count = len(subjects_to_download)
    print(f"📦 Successfully parsed {total_count} international subject records.")
    print("🚀 Initiating unrestricted background download pipeline...")
    
    # Base URL for the CPAC Harvard-Oxford (ho) 90-region pipeline files
    base_s3_url = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_noglobal/rois_ho/"
    
    success_count = 0
    fail_count = 0

    # REMOVED LIMIT: Loops through all 1,000+ entries systematically
    for idx, (site, sub_id) in enumerate(subjects_to_download):
        file_name = f"{site}_00{sub_id}_rois_ho.1D"
        target_url = f"{base_s3_url}{file_name}"
        dest_path = os.path.join(data_dir, file_name)
        
        # Skip if the patient matrix is already sitting on your hard drive
        if os.path.exists(dest_path):
            success_count += 1
            continue
            
        try:
            urllib.request.urlretrieve(target_url, dest_path)
            success_count += 1
            if success_count % 10 == 0:
                print(f"📈 Progress: Loaded [{success_count + fail_count}/{total_count}] matrices safely into local storage.")
        except Exception:
            # Fallback handling: Try matching alternate naming protocols if scanning labels diverge
            try:
                # Sometimes sub IDs are zero-padded differently depending on scanning hubs
                alt_file = f"{site}_{sub_id}_rois_ho.1D"
                urllib.request.urlretrieve(f"{base_s3_url}{alt_file}", os.path.join(data_dir, alt_file))
                success_count += 1
            except Exception:
                fail_count += 1
                continue

    print(f"\n🏁 Finished! Successfully loaded: {success_count} files. Missing/Corrupted rows skipped: {fail_count}.")

if __name__ == "__main__":
    download_complete_abide_dataset()