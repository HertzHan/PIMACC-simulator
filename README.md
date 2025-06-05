# PIMACC

PIMACC is an end-to-end CIM accuracy simulator, which works with mappings and instructions given by CIM compiler. PIMACC is devoted to simulator non-ideal effects in CIM inference, such as quantization error, error of resistance, IR-drop effects, etc. Error injections are flexible and can be configured in hardware configuration file.

PIMACC is based on PIMCOMP-NN, an CIM compiler, link given hear: https://github.com/sunxt99/PIMCOMP-NN. To run PIMACC, you need to compile your target NN network with PIMCOMP-NN, including front end and back end. To make sure you can run verification after compiling, use `–v=YES` in backend compiling

# Usage

## Requirements

In order to run the PIMACC verification program, you need to install the following python package.

- onnx
- onnxruntime
- numpy
- torch
- torchvision
- cv2

## Simulation
Before running the simulator, you need to make sure you have the same onnx model before compilation, and the result in `PIMACC/output/VerificationInfo.json` after compilation.

```shell
cd PIMACC/verification/
python verification.py --model_path ../models/ONNX/resnet18.onnx --pipeline_type element --image_num 1000 --batchsize 10
```
Simuilator will run and give the accuracy result

## configaration
Non-ideal effects are configured in PIMACC/config.json, you can set values of different effects.
- "cell_precision": cell precision
- "reference_conductance_state": the resistance of low-resistance-state
- "R_ratio": on-off ration, the ratio of high and low resistances
- "bitline_resistance": wire resistance between devices on bitline
- "wordline_resistance":wire resistance between devices on bitline
- "variation": device resistance variation, in form of standard deviation
- "SAF_flag": whether to consider SAF fault
- "p_SA0": probability of SA0
- "p_SA1": probability of SA1

