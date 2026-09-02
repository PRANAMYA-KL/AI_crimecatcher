import torch
import argparse
import time

# Correct import based on your folder structure:
from Fight_Detection_Pip_Package.fight_detection.Fight_utils import loadModel, predict_on_video, start_streaming

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch Fight Detection Inference')
parser.add_argument('--modelPath', required=True, help="Path to the .pth model file")
parser.add_argument('--streaming', action='store_true', help="Set this flag to use webcam or IP camera")
parser.add_argument('--inputPath', default=None, help="Path to video file, stream URL, or webcam index (use 0 for default webcam)")
parser.add_argument('--outputPath', default=None, help="Path to save output video (optional)")
parser.add_argument('--sequenceLength', type=int, default=16, help="Frames per sequence for temporal model")
parser.add_argument('--skip', type=int, default=2, help="Frame skip interval")
parser.add_argument('--showInfo', action='store_true', help="Show debug/info during inference")

def main():
    args = parser.parse_args()

    model = loadModel(args.modelPath)
    
    if args.streaming:
        # Use webcam by default if no inputPath is given
        stream_source = args.inputPath if args.inputPath is not None else 0
        print(f"[INFO] Starting real-time streaming detection from: {stream_source}")
        start_streaming(model, stream_source)
    else:
        start = time.time()
        print(f"[INFO] Running fight detection on video: {args.inputPath}")
        predict_on_video(
            args.inputPath,
            args.outputPath,
            model,
            args.sequenceLength,
            args.skip,
            args.showInfo
        )
        end = time.time()
        print(f"Processing time: {end-start:.2f}s")

if __name__ == '__main__':
    main()
