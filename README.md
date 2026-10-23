## Abstract
We attempt to map Bharatanatiyam steps to any Indian classical song on the basis of nritta only i.e. rhythm and tempo. We use librosa and Demucs to analyse the song andconvert it into chunks with beat information and then map the adavus to the beats.

## How to initialize and run?
- Initialize and sync the project with uv. 
- To run the script:- uv run pipeline.py <your-song> --adavus landmarks_json -o output --tala 8 --default-beats 4
- The output folder will have the index.html file



 
