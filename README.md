# DSAMNet PS/PL Codesign + DBWFB2 Wavelet Filter on ZedBoard

PS/PL hardware-software co-design of DSAMNet for remote-sensing change detection on the Xilinx Zynq-7000 ZedBoard, integrating a hardware-accelerated DBWFB2 wavelet LL-subband filter.

---

## 1. Project Overview

This project demonstrates the deployment and hardware-software co-design of **DSAMNet (Deeply Supervised Attention Metric-Based Network)** for remote-sensing image change detection on a **Digilent ZedBoard based on the Xilinx Zynq-7000 SoC**.

The design uses both sides of the Zynq device:

- **Processing System (PS):** ARM Cortex-A9
- **Programmable Logic (PL):** FPGA fabric

The project was developed in two major stages:

1. **Baseline DSAMNet PS/PL Codesign**
2. **DSAMNet + DBWFB2 Wavelet Filter Codesign**

The baseline establishes DSAMNet inference on the ZedBoard and accelerates CBAM in PL using Vitis HLS.

The extended implementation adds a hardware DBWFB2 wavelet filter before DSAMNet. The current integration focuses on the **LL (low-low) subband**, producing a 3 × 128 × 128 representation from a 3 × 256 × 256 input.

---

# 2. Original Software, Research and References

This project builds on existing research and software implementations. The original projects are referenced rather than copied wholesale into this repository.

## 2.1 DSAMNet Paper

**Title:**

*A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing Change Detection*

**Publication:**

IEEE Transactions on Geoscience and Remote Sensing, 2021.

The paper provides the DSAMNet architecture and the remote-sensing change-detection methodology used as the algorithmic reference for this project.

---

## 2.2 Original DSAMNet PyTorch Implementation

Original DSAMNet repository:

https://github.com/liumency/DSAMNet

This repository provides the original PyTorch implementation used as the software/reference baseline for this project.

The hardware-software implementation in this repository should be distinguished from the original DSAMNet work.

---

## 2.3 SYSU-CD Dataset / Reference Repository

SYSU-CD repository:

https://github.com/liumency/SYSU-CD

The SYSU-CD repository provides the dataset/reference resources associated with remote-sensing change detection.

The dataset itself is not redistributed in this repository.

---

## 2.4 DBWFB2 Reference Repository

DBWFB2 reference repository:

https://github.com/limitbreaker9/DBWFB2-Zynq-INT8-CNN

This project was used as a reference for the DBWFB2 FPGA wavelet-filter architecture and implementation approach.

The DBWFB2 filter in this repository is integrated into the DSAMNet PS/PL codesign flow and adapted for the ZedBoard-based system.

---

## 2.5 Reference vs. Project Implementation

### Original / Reference Work

- DSAMNet neural-network architecture
- Original DSAMNet PyTorch implementation
- SYSU-CD dataset/reference resources
- DBWFB2 reference hardware architecture

### Work Implemented / Integrated in This Project

- Zynq-7000 PS/PL DSAMNet deployment
- ZedBoard integration
- PS/PL partitioning
- CBAM hardware acceleration using Vitis HLS
- Fixed-point PS/PL data conversion
- AXI-based PS/PL communication
- DBWFB2 LL-subband hardware integration
- DBWFB2 custom RTL integration
- AXI DMA image transfer
- PS-side LL preprocessing
- DSAMNet + DBWFB2 integration
- Python reference and validation infrastructure
- RTL and hardware validation
- ZedBoard execution and performance evaluation
- Multiple DSAMNet processing modes

---

# 3. Key Contributions

The main engineering contributions documented in this repository are:

- Deployment of DSAMNet on a Zynq-7000 ZedBoard.
- Partitioning of the DSAMNet pipeline between ARM PS and FPGA PL.
- CBAM acceleration using Vitis HLS.
- Fixed-point representation for PS/PL CBAM data exchange.
- AXI-Lite control and AXI memory-mapped data movement.
- Integration of a custom DBWFB2 LL wavelet filter in PL.
- AXI DMA based image transfer between DDR and PL.
- PS-side preprocessing and integration of LL output with DSAMNet.
- Custom Verilog RTL for the DBWFB2 streaming datapath.
- Python reference models and golden-vector generation.
- Cycle-level, RTL-level and board-level validation.
- Bit-exact comparison of hardware DBWFB2 output with the Python reference.
- Baseline, LL+upsample and LL+no-pool experimental modes.
- FPGA timing and resource evaluation.

---

# 4. System Architecture

## 4.1 Overall Architecture

```text
                 Remote-Sensing Image
                         │
                         ▼
                ┌──────────────────┐
                │   DBWFB2 Filter  │
                │       (PL)       │
                └────────┬─────────┘
                         │
                         ▼
                 LL Subband Output
                    3 × 128 × 128
                         │
                         ▼
              PS-side Preprocessing
                         │
                         ▼
                ┌──────────────────┐
                │     DSAMNet      │
                │       (PS)       │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ CBAM Accelerator │
                │       (PL)       │
                └────────┬─────────┘
                         │
                         ▼
                 Feature Distance
                         │
                         ▼
                 Change Detection
                         │
                         ▼
                    Change Map
```

---

# 5. Hardware Platform

| Component | Configuration |
|---|---|
| Development Board | Digilent ZedBoard |
| SoC | Xilinx Zynq-7000 |
| Processing System | ARM Cortex-A9 |
| FPGA Fabric | Zynq-7000 Programmable Logic |
| Design Tool | Vivado |
| Software Environment | Vitis |
| HLS Tool | Vitis HLS |
| Hardware Description | Verilog RTL + HLS C/C++ |
| Communication | AXI / AXI-Lite |
| Data Transfer | AXI DMA |
| Memory | DDR |

---

# 6. Baseline DSAMNet PS/PL Codesign

The baseline implementation is located under:

```text
01_Baseline_Codesign/
└── DSAMNet_Codesign/
```

The baseline establishes the DSAMNet inference pipeline on the ZedBoard.

## 6.1 Processing System

The ARM Cortex-A9 executes the major DSAMNet operations:

- Backbone
- Decoder
- Feature-distance computation
- Post-processing
- Change-map generation

Important PS application files include:

```text
DSAMNet_App/src/
├── ps_backbone.cpp
├── ps_cbam_hw.cpp
├── ps_cbam_hw.h
├── ps_decoder.cpp
├── ps_dsamnet.h
├── ps_main.cpp
└── ps_postprocess.cpp
```

A separate software implementation/reference directory is also retained under:

```text
PS_Software/
```

---

# 7. CBAM Hardware Acceleration

CBAM is accelerated in the FPGA Programmable Logic using **Vitis HLS**.

The HLS implementation is located under:

```text
01_Baseline_Codesign/
└── DSAMNet_Codesign/
    └── PL_CBAM_HLS/
```

Important files include:

```text
cbam_accel.cpp
cbam_accel.h
tb_cbam_accel.cpp
hls_config.cfg
dfx_runtime.txt
```

The accelerator provides:

- AXI-Lite control
- AXI memory-mapped input access
- AXI memory-mapped weight access
- AXI memory-mapped output access
- Fixed-point feature processing

The PS software prepares the accelerator inputs, manages cache coherency, starts the accelerator and converts the output back to the PS representation.

---

# 8. Fixed-Point PS/PL Interface

The PL CBAM implementation uses fixed-point arithmetic.

The baseline PS/PL flow is:

```text
PS Floating-Point Feature Map
            │
            ▼
      Fixed-Point Conversion
            │
            ▼
         DDR Memory
            │
            ▼
      AXI Memory Interface
            │
            ▼
       CBAM Accelerator
            │
            ▼
      Fixed-Point Output
            │
            ▼
      PS Floating-Point
            │
            ▼
       DSAMNet Pipeline
```

Cache flush/invalidate operations are used to maintain consistency between PS-side memory and PL-side AXI accesses.

---

# 9. Baseline Vivado Hardware Design

The baseline Vivado project is located under:

```text
01_Baseline_Codesign/
└── DSAMNet_Codesign/
    └── DSAMNet_Vivado/
```

The retained hardware design includes the main block design, IP configuration and project sources.

Major components include:

- Processing System 7
- CBAM HLS accelerator
- AXI SmartConnect
- AXI memory interconnect
- DDR interface
- System ILA

Generated implementation artifacts such as bitstreams, XSA files, synthesis databases and implementation runs are intentionally excluded.

---

# 10. DBWFB2 Wavelet Codesign

The extended implementation is located under:

```text
02_DBWFB2_Wavelet_Codesign/
└── DSAMNet_Filter_Codesign/
```

The DBWFB2 filter is inserted before DSAMNet.

For a 3 × 256 × 256 input:

```text
Input
3 × 256 × 256
      │
      ▼
DBWFB2 LL Filter (PL)
      │
      ▼
LL Output
3 × 128 × 128
```

The current implementation focuses on the LL subband.

---

# 11. DBWFB2 Hardware Architecture

The DBWFB2 filter is implemented as a streaming hardware datapath in Verilog.

The architecture consists of:

```text
                  DBWFB2 LL Engine
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    Line Buffers   Vertical Filter   Horizontal Filter
                                         │
                                         ▼
                                  Output Normalization
```

Important RTL files:

```text
DBWFB2_Vivado/
└── DBWFB2_Vivado.srcs/
    └── sources_1/
        └── new/
            ├── dbwfb2_fir9.v
            ├── dbwfb2_linebuf.v
            └── dbwfb2_ll_engine.v
```

A dedicated simulation testbench is included:

```text
DBWFB2_Vivado.srcs/sim_1/new/tb_dbwfb2_ll_engine.v
```

---

# 12. DBWFB2 Processing Characteristics

The implemented LL filter processes the input in planar CHW form.

The hardware performs:

- Vertical 9-tap filtering
- Horizontal 9-tap filtering
- Line-buffer based streaming
- Replicate boundary handling
- Even-row/even-column LL extraction
- Fixed-point rounding
- Output clipping to the int8 range

The tested full-image configuration is:

```text
Input  : 3 × 256 × 256
Output : 3 × 128 × 128
```

The filter produces one output token per clock when the pipeline is not stalled.

Output backpressure causes the streaming pipeline to stall globally.

---

# 13. AXI DMA Integration

The DBWFB2 filter uses AXI DMA to transfer image data between PS DDR memory and PL.

```text
                 ARM PS
                   │
                   ▼
              DDR Memory
                   │
                   │ MM2S
                   ▼
                AXI DMA
                   │
                   ▼
            DBWFB2 Filter
                 (PL)
                   │
                   │ S2MM
                   ▼
                AXI DMA
                   │
                   ▼
              DDR Memory
                   │
                   ▼
                 ARM PS
```

The PS-side DMA wrapper is implemented in:

```text
DBWFB2_App/src/
├── ps_filter_hw.cpp
└── ps_filter_hw.h
```

The software handles:

- DMA initialization
- Input/output buffer management
- Cache flush
- S2MM setup
- MM2S setup
- DMA polling
- Cache invalidation
- Output copy
- Timing measurement

---

# 14. DSAMNet Processing Modes

The DBWFB2-integrated implementation provides three processing modes.

## Mode 0 — Baseline

No DBWFB2 preprocessing.

```text
3 × 256 × 256
       │
       ▼
    DSAMNet
       │
       ▼
Change Detection
```

Mode 0 serves as the baseline for comparison.

---

## Mode 1 — LL + Upsample

The DBWFB2 hardware produces:

```text
3 × 128 × 128
```

The PS then upsamples the LL output to 256 × 256 before running the original DSAMNet input path.

```text
3 × 256 × 256
       │
       ▼
DBWFB2 LL Filter
       │
       ▼
3 × 128 × 128
       │
       ▼
PS Upsampling
       │
       ▼
3 × 256 × 256
       │
       ▼
DSAMNet
```

---

## Mode 2 — LL + No Pool

The LL output remains at 128 × 128.

The DSAMNet front-end is modified to accept the reduced spatial resolution without the original maximum-pooling stage.

```text
3 × 256 × 256
       │
       ▼
DBWFB2 LL Filter
       │
       ▼
3 × 128 × 128
       │
       ▼
Modified DSAMNet Stem
       │
       ▼
DSAMNet
```

Mode 2 is a modified network path and is not numerically equivalent to the original DSAMNet architecture.

---

# 15. DBWFB2 PS Application

The DBWFB2 application is located under:

```text
DBWFB2_App/src/
```

Important files include:

```text
ps_backbone.cpp
ps_bench.h
ps_cbam_hw.cpp
ps_cbam_hw.h
ps_conv_fast.h
ps_decoder.cpp
ps_dsamnet.h
ps_filter_hw.cpp
ps_filter_hw.h
ps_ll_prep.cpp
ps_main.cpp
ps_postprocess.cpp
ps_timer.h
```

Responsibilities include:

- DSAMNet inference
- DBWFB2 DMA control
- LL preprocessing
- Timing
- CBAM hardware interaction
- Decoder
- Post-processing
- Change detection

---

# 16. Python Reference and Validation Infrastructure

Python reference implementations are located under:

```text
02_DBWFB2_Wavelet_Codesign/
└── DSAMNet_Filter_Codesign/
    └── reference/
```

Important files include:

```text
accuracy_check.py
dbwfb2_ll_ref.py
ll_variants.py
make_cbam_goldens.py
make_ll_golden.py
```

Reference vectors are stored under:

```text
reference/vectors/
```

The vectors include 32 × 32 and 256 × 256 validation data.

This infrastructure provides a software reference against which the RTL and board implementations can be compared.

---

# 17. Validation Methodology

Validation was performed at multiple levels.

## 17.1 Python Reference Validation

Multiple independent Python implementations were compared to establish consistency of the DBWFB2 LL computation.

The reference implementation uses the same filter constants and integer processing behavior as the hardware design.

---

## 17.2 Cycle-Level Validation

A cycle-level model was used to validate the streaming behavior of the hardware design.

The validation included:

- Random input stalls
- Output backpressure
- Non-square planes
- Multiple images processed back-to-back
- Full 3 × 256 × 256 input

The cycle-level model matched the reference computation bit-for-bit for the tested configurations.

---

## 17.3 RTL Simulation

Vivado XSim was used to verify the DBWFB2 RTL.

Test cases included:

### Small test

```text
3 × 32 × 32
2 images
```

Results:

```text
1536 outputs
0 value errors
0 TLAST errors
```

### Full-image test

```text
3 × 256 × 256
1 image
```

Results:

```text
49152 outputs
0 value errors
0 TLAST errors
```

Random input gaps and output backpressure were also exercised.

---

# 18. ZedBoard Hardware Validation

The complete DBWFB2 PS/PL path was executed on the ZedBoard.

The board-level flow verifies:

```text
Input Image
    ↓
PS DDR
    ↓
AXI DMA
    ↓
DBWFB2 PL
    ↓
AXI DMA
    ↓
PS DDR
    ↓
Python Reference Comparison
```

For the tested full-image configuration:

```text
Input  : 3 × 256 × 256
Output : 3 × 128 × 128
Output : 49152 bytes
```

The ZedBoard execution produced:

```text
LL vs Python reference:
0 mismatches of 49152
```

This demonstrates bit-exact hardware/reference agreement for the tested LL output.

---

# 19. Baseline CBAM Validation

The baseline CBAM accelerator was also validated against the corresponding software/reference path.

The fixed-point interface was checked for saturation and numerical consistency.

The validated PS/PL implementation achieved high correlation with the reference distance map after correcting the fixed-point data representation.

The final baseline validation included:

```text
MAE  = 0.201471
RMSE = 0.219470
Corr = 0.988358
```

These values represent numerical similarity against the reference distance map and should not be interpreted as change-detection accuracy metrics.

---

# 20. Mode 1 ZedBoard Execution

A complete Mode 1 execution was performed on the ZedBoard.

Configuration:

```text
Mode 1:
PL DBWFB2 filter
        ↓
LL 3 × 128 × 128
        ↓
PS upsample
        ↓
DSAMNet
```

The hardware filter validation during this execution reported:

```text
T1 filter:
~1.994 ms DMA/filter measurement
0 mismatches of 49152

T2 filter:
~1.995 ms DMA/filter measurement
0 mismatches of 49152
```

The end-to-end timing for the tested run was approximately:

```text
Backbone : 304.40 s
Decoder  : 239.69 s
CBAM     : 0.299 s
Total    : ~1089.1 s
         : ~18.2 minutes
```

The DBWFB2 filter itself is therefore not the dominant runtime component in the current end-to-end implementation.

The major runtime contributors remain the PS-side DSAMNet backbone and decoder.

---

# 21. Change Detection Result

The tested Mode 1 execution produced:

```text
Changed pixels : 42550 / 65536
               : 64.93 %

Distance mean  : 1.1502
Distance min   : 0.0177
Distance max   : 3.0418
```

The pipeline detected a change between the tested image pair.

A comparison against the unfiltered PyTorch distance map produced approximately:

```text
MAE     : 0.391234
RMSE    : 0.503483
Max Err : 1.519057
Corr    : 0.656396
```

These values measure similarity between distance maps.

They are **not** ground-truth change-detection accuracy metrics.

For a proper accuracy study, the filtered pipeline should be evaluated using ground-truth metrics such as F1, IoU, precision and recall.

---

# 22. Hardware Timing and Resource Results

The DBWFB2 hardware implementation was synthesized and implemented in Vivado.

The tested implementation achieved timing closure.

Reported timing results:

```text
WNS          : +0.611 ns
TNS          : 0
WHS          : +0.022 ns
THS          : 0
Failed Routes: 0
```

Reported resource usage:

```text
LUT  : 15814
FF   : 20012
BRAM : 29
DSP  : 37
URAM : 0
```

Reported implementation power:

```text
~2.011 W
```

These values correspond to the tested hardware configuration and can vary with tool versions, constraints and implementation settings.

---

# 23. Important Performance Interpretation

The current implementation should not be described as demonstrating a proportional 4× reduction in complete DSAMNet computation simply because the DBWFB2 LL output has half the height and width.

The current modes have different computational behavior:

### Mode 1

The LL result is upsampled back to 256 × 256.

Therefore, most of the original DSAMNet spatial computation remains.

### Mode 2

The network operates on the reduced spatial resolution at the front end, but the architecture is modified and does not currently represent a complete native low-resolution DSAMNet redesign.

Therefore, the present work demonstrates:

- Hardware wavelet preprocessing
- PS/PL integration
- Reduced-resolution feature generation
- Hardware validation
- End-to-end system integration

rather than claiming a theoretical full-network 4× computational reduction.

---

# 24. Accuracy Considerations

Using only the LL subband introduces an information trade-off.

The LL subband preserves the lower-frequency/global structure of the input but discards high-frequency information represented by the LH, HL and HH subbands.

This does **not** mean that accuracy cannot be improved.

Potential ways to improve the change-detection accuracy include:

- Fine-tuning DSAMNet using LL-filtered training data
- Retraining the network for wavelet-domain inputs
- Combining LL with selected high-frequency subbands
- Adaptive subband selection
- Learned subband fusion
- Native low-resolution network design
- Ground-truth-driven optimization

High-frequency subbands may contain useful structural information such as edges and fine changes, which can be important for remote-sensing change detection.

---

# 25. Repository Structure

```text
DSAMNet-PSPL-Codesign-DBWFB2-ZedBoard/
│
├── README.md
├── .gitignore
│
├── 01_Baseline_Codesign/
│   └── DSAMNet_Codesign/
│       │
│       ├── DSAMNet_App/
│       │   └── src/
│       │
│       ├── DSAMNet_Platform/
│       │
│       ├── DSAMNet_Vivado/
│       │
│       ├── PL_CBAM_HLS/
│       │
│       └── PS_Software/
│
├── 02_DBWFB2_Wavelet_Codesign/
│   └── DSAMNet_Filter_Codesign/
│       │
│       ├── DBWFB2_App/
│       │   └── src/
│       │
│       ├── DBWFB2_Platform/
│       │
│       ├── DBWFB2_Vivado/
│       │
│       ├── PL_CBAM_HLS/
│       │
│       └── reference/
│           └── vectors/
│
├── references/
│   └── README.md
│
└── docs/
```

---

# 26. Important Files

## Baseline PS Application

```text
DSAMNet_App/src/
├── ps_backbone.cpp
├── ps_cbam_hw.cpp
├── ps_cbam_hw.h
├── ps_decoder.cpp
├── ps_dsamnet.h
├── ps_main.cpp
└── ps_postprocess.cpp
```

## Baseline CBAM HLS

```text
PL_CBAM_HLS/
├── cbam_accel.cpp
├── cbam_accel.h
├── tb_cbam_accel.cpp
└── hls_config.cfg
```

## DBWFB2 PS Application

```text
DBWFB2_App/src/
├── ps_backbone.cpp
├── ps_bench.h
├── ps_cbam_hw.cpp
├── ps_cbam_hw.h
├── ps_conv_fast.h
├── ps_decoder.cpp
├── ps_dsamnet.h
├── ps_filter_hw.cpp
├── ps_filter_hw.h
├── ps_ll_prep.cpp
├── ps_main.cpp
├── ps_postprocess.cpp
└── ps_timer.h
```

## DBWFB2 RTL

```text
DBWFB2_Vivado/
└── DBWFB2_Vivado.srcs/
    ├── sim_1/new/
    │   └── tb_dbwfb2_ll_engine.v
    │
    └── sources_1/new/
        ├── dbwfb2_fir9.v
        ├── dbwfb2_linebuf.v
        └── dbwfb2_ll_engine.v
```

## DBWFB2 Reference

```text
reference/
├── accuracy_check.py
├── dbwfb2_ll_ref.py
├── ll_variants.py
├── make_cbam_goldens.py
├── make_ll_golden.py
└── vectors/
    ├── exp_256.hex
    ├── exp_32.hex
    ├── expv_256.hex
    ├── expv_32.hex
    ├── in_256.hex
    └── in_32.hex
```

---

# 27. Reproducibility

The repository retains the important source and hardware-design files needed to understand and reproduce the implementation:

- C/C++ PS software
- HLS accelerator source
- HLS configuration
- Verilog RTL
- RTL testbench
- Vivado block designs
- Vivado IP configuration
- Vitis application configuration
- Python reference implementations
- Golden vectors
- Validation scripts
- Project-specific configuration files

Generated artifacts are intentionally excluded, including:

- `.elf`
- `.bit`
- `.bin`
- `.xsa`
- `.dcp`
- Vivado synthesis/implementation runs
- Vivado generated databases
- Vitis generated build output
- BSP-generated content
- HLS generated reports
- logs
- temporary workspace files
- datasets
- recorded videos
- paper PDFs

A fresh reproduction should use compatible Vivado, Vitis and Vitis HLS versions and the same ZedBoard/Zynq-7000 target configuration.

---

# 28. Build and Deployment Flow

The overall development flow is:

```text
Original DSAMNet
      │
      ▼
PyTorch Software Reference
      │
      ▼
Algorithm / Layer Analysis
      │
      ▼
PS/PL Partitioning
      │
      ├──────────────────────┐
      ▼                      ▼
PS C/C++ Implementation    PL Accelerator
      │                      │
      │                HLS / RTL
      │                      │
      └──────────┬───────────┘
                 ▼
          Vivado Integration
                 │
                 ▼
          AXI / DDR / DMA
                 │
                 ▼
          Vitis Application
                 │
                 ▼
             ZedBoard
                 │
                 ▼
          Hardware Validation
                 │
                 ▼
       Performance Evaluation
```

---

# 29. Limitations

The current implementation has the following limitations:

1. The current wavelet integration focuses on the LL subband.
2. LH, HL and HH subbands are not yet integrated into the DSAMNet inference path.
3. Mode 1 upsamples the LL result back to the original spatial resolution.
4. Mode 2 modifies the DSAMNet front-end and is not equivalent to the original network.
5. The PS-side backbone and decoder remain the major runtime bottlenecks.
6. Ground-truth-based accuracy evaluation is required for a complete change-detection study.
7. The current implementation does not demonstrate a full-network 4× computational reduction.
8. Further optimization is required for practical end-to-end FPGA acceleration.

---

# 30. Future Work

Potential extensions include:

- Fine-tuning DSAMNet with LL-filtered training data.
- Evaluating LL, LH, HL and HH subbands independently.
- Combining multiple wavelet subbands.
- Adaptive or learned subband selection.
- Learned wavelet-feature fusion.
- Native 128 × 128 DSAMNet architecture.
- Rebuilding CBAM for reduced-resolution operation.
- Additional PS/PL partitioning.
- Hardware acceleration of DSAMNet backbone layers.
- Hardware acceleration of decoder layers.
- DDR bandwidth optimization.
- AXI DMA optimization.
- Reduced-precision inference.
- Quantization-aware training.
- End-to-end latency optimization.
- Ground-truth-based F1/IoU/precision/recall evaluation.
- FPGA resource versus accuracy trade-off analysis.

---

# 31. Technologies Used

## Hardware

- Digilent ZedBoard
- Xilinx Zynq-7000
- ARM Cortex-A9
- FPGA Programmable Logic
- DDR memory

## FPGA / Hardware Design

- AMD/Xilinx Vivado
- Vitis
- Vitis HLS
- Verilog
- AXI
- AXI-Lite
- AXI DMA
- AXI memory-mapped interfaces
- System ILA

## Software

- C
- C++
- Python
- PyTorch
- CMake

## Algorithms

- DSAMNet
- CBAM
- DBWFB2 wavelet filtering
- Remote-sensing image change detection

---

# 32. Original Software Attribution

The original DSAMNet architecture and software implementation are attributed to their original authors.

DSAMNet:

https://github.com/liumency/DSAMNet

SYSU-CD:

https://github.com/liumency/SYSU-CD

DBWFB2 reference:

https://github.com/limitbreaker9/DBWFB2-Zynq-INT8-CNN

This repository does not redistribute the original paper or dataset.

Please refer to the original repositories and publications for their respective licenses and usage conditions.

---

# 33. Project Scope

The purpose of this repository is to document the complete hardware-software co-design path from a software neural-network reference to FPGA-based deployment.

The project covers:

```text
Software Reference
       ↓
Network Analysis
       ↓
PS/PL Partition
       ↓
HLS / RTL Development
       ↓
Vivado Hardware Integration
       ↓
AXI / DMA Integration
       ↓
Vitis PS Software
       ↓
ZedBoard Deployment
       ↓
Hardware Validation
       ↓
Performance Evaluation
```

The repository is intended to provide a clear record of the architecture, implementation, validation methodology and engineering decisions involved in the project.

---

# 34. Author

**Rajasri Kutikuppala**

B.Tech – Electronics and Communication Engineering  
IIIT Sri City

---

## Disclaimer

This repository is intended for academic, research and engineering portfolio purposes.

The original DSAMNet implementation, paper and datasets remain attributed to their respective authors. The external repositories linked above should be used for the original software, publications and datasets.

This repository documents the PS/PL hardware-software codesign, FPGA integration, DBWFB2 hardware implementation, validation and ZedBoard deployment work associated with this project.
