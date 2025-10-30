import os

from torch.utils.data import Dataset


class VimSketchDataset(Dataset):
    def __init__(self, root_dir):
        """
        Dataset class for the VimSketch Dataset.
        Audio files: 
            Format: .WAV (16-bit PCM)
            Sampling rate: 44.1 kHz 
            Number of channels: 1
        Args:
            root_dir (str): Path to the directory containing:
                - reference_file_names.csv
                - vocal_imitation_file_names.csv
                - references (folder with reference files)
                - vocal_imitations (folder with imitation files)
        """
        self.root_dir = root_dir

        # Load filenames from CSV files (each line is a filepath, but only the filename is given)
        with open(os.path.join(root_dir, "reference_file_names.csv"), "r") as f:
            self.references = [x.strip() for x in f.readlines()]
        with open(os.path.join(root_dir, "vocal_imitation_file_names.csv"), "r") as f:
            self.vocal_imitations = [x.strip() for x in f.readlines()]

        # Create mapping between references and vocal imitations
        self.mapping = {}
        for reference_idx, reference in enumerate(self.references):
            # get description from reference filename
            if reference[7] == "-":  # Vocal Sketch Dataset
                description = reference.split("-", 1)[1]
                # extract audio class and text, subclass is empty
                # remove .wav, split at "-" and replace "_" with " "
                string_parts = description.split(".")[0].split("-")
                audio_class = string_parts[1].replace("_", " ")
                text = string_parts[0].replace("_", " ")
                subclasses = []
            else:  # Vocal Imitation Dataset
                description = reference[7:]
                # extract audio class, text, and subclasses
                # remove .wav, split at "_"
                string_parts = description.split(".")[0].split("_")
                audio_class = string_parts[0]
                text = string_parts[-1]
                subclasses = string_parts[1:-1]

            for imitation_idx, imitation in enumerate(self.vocal_imitations):
                # get description from imitation filename
                if imitation[9] == "-":  # Vocal Sketch Dataset
                    imitation_description = imitation.split("-", 1)[1]
                else:  # Vocal Imitation Dataset
                    imitation_description = imitation[9:]

                # check if descriptions match
                if description == imitation_description:
                    if reference_idx not in self.mapping:
                        self.mapping[reference_idx] = ([], [audio_class, text, subclasses])
                    self.mapping[reference_idx][0].append(imitation_idx)

        # Invert mapping: key is imitation index, value is (reference index, (audio_class, text, subclasses))
        self.inverted_mapping = {}
        for reference_idx, (imitation_indices, class_info) in self.mapping.items():
            for imitation_idx in imitation_indices:
                self.inverted_mapping[imitation_idx] = (reference_idx, class_info)

        # Create a list of valid imitation indices for dataset access
        self.valid_imitation_indices = list(self.inverted_mapping.keys())

    def __len__(self):
        return len(self.valid_imitation_indices)

    def __getitem__(self, idx):
        # Get the actual imitation index from the valid list
        imitation_idx = self.valid_imitation_indices[idx]
        reference_idx, (audio_class, text, subclasses) = self.inverted_mapping[
            imitation_idx
        ]

        # Adjust paths: prepend the folder names to the filenames
        imitation_path = os.path.join(
            self.root_dir, "vocal_imitations", self.vocal_imitations[imitation_idx]
        )
        reference_path = os.path.join(
            self.root_dir, "references", self.references[reference_idx]
        )

        # create text with audio class, subclasses, and text
        sub = "".join([', ' + s for s in subclasses])
        text = f"{audio_class}{sub}, {text}"
       
        return {
            "imitation_path": imitation_path,
            "reference_path": reference_path,
            "text": text,
        }
    
    def get_references(self):
        # iterate over references using self.mapping
        # yields reference path and text
        for reference_idx, (imitation_indices, (audio_class, text, subclasses)) in self.mapping.items():
            reference_path = os.path.join(
                self.root_dir, "references", self.references[reference_idx]
            )
            sub = "".join([', ' + s for s in subclasses])
            text = f"{audio_class}{sub}, {text}"
            yield {
                "reference_path": reference_path,
                "text": text
            }
    

class VimSketchDatasetGenerated(VimSketchDataset):

    def __init__(self, root_dir, generated_dir):
        """
        Dataset class for the VimSketch Dataset with generated audio files.
        Audio files: 
            Format: .WAV (16-bit PCM)
            Sampling rate: 44.1 kHz 
            Number of channels: 1
        Args:   root_dir (str): Path to the directory containing:
                - reference_file_names.csv
                - vocal_imitation_file_names.csv
                - references (folder with reference files)
                - vocal_imitations (folder with imitation files)
            generated_dir (str): Path to the directory containing the generated audio files
        """
        super().__init__(root_dir)
        self.generated_dir = generated_dir # e.g. "style_transfer/transfer_strength_0.5"

    def __getitem__(self, idx):
        # get stuff from parent class
        data = super().__getitem__(idx)
        # adjust filename
        imitation_filename = self.vocal_imitations[idx]
                
        # add generated audio path
        generated_path = os.path.join(self.root_dir, self.generated_dir, imitation_filename)
        data["generated_path"] = generated_path
        return data




if __name__ == "__main__":
    # test dataset by loading all files once
    # create logfile
    import logging

    import soundfile as sf
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    logging.basicConfig(
        filename="dataset.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Loading dataset...")
    dataset = VimSketchDataset("../dataset/Vim_Sketch_Dataset")

    # print some stats about the dataset
    print(f"Number of references: {len(dataset.references)}")
    print(f"Number of vocal imitations: {len(dataset.vocal_imitations)}")
    print(f"Number of valid imitations: {len(dataset.valid_imitation_indices)}")
    print(f"Length of Dataset: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

  
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        
        logging.info(f"Loading Batch {batch_idx}")
    
        for i in range(len(batch["imitation_path"])):
            logging.info(f"Loading files {i}...")
            imitation_path = batch["imitation_path"][i]
            reference_path = batch["reference_path"][i]
            text = batch["text"][i]
            
            # write text to txt file
            with open("texts.txt", "a") as f:
                f.write(f"{text}\n")

            # # load the audio files
            # imitation, _ = sf.read(imitation_path)
            # reference, _ = sf.read(reference_path)
            # # get the length of the audio files
            # imitation_length = len(imitation) / 44100
            # reference_length = len(reference) / 44100
            # logging.info("\n"
            #     f"Imitation: {imitation_path} ({imitation_length:.2f}s),\n"
            #     f"Reference: {reference_path} ({reference_length:.2f}s),\n"
            #     f"Text: {text}"
            # )
    

    # get unique texts by iterating over the dataset.mapping
    for reference_idx, (imitation_indices, (audio_class, text, subclasses)) in dataset.mapping.items():
        # create text with audio class, subclasses, and text
        sub = "".join([', ' + s for s in subclasses])
        text = f"{audio_class}{sub}, {text}"
        # write text to txt file
        with open("unique_texts.txt", "a") as f:
            f.write(f"{text}\n")

        
            


        