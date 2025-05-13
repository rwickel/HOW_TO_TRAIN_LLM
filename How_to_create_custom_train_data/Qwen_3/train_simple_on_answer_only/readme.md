# Fine-tuning Qwen Model for ISO 21448 (SOTIF) Question Answering Example

## Overview

This project provides a Python script to fine-tune a pre-trained Qwen language model (specifically "Qwen/Qwen3-4B") for question answering based on a dataset related to the ISO 21448 standard (Safety of the Intended Functionality - SOTIF). The script leverages the Hugging Face `transformers` library for model loading and training, and the `datasets` library for data handling.

The process includes:
1.  Loading the "Qwen/Qwen3-4B" model and its tokenizer.
2.  Preparing a custom dataset with question-answer pairs about ISO 21448.
3.  Formatting the dataset into the Qwen chat template.
4.  Tokenizing the formatted data for causal language modeling.
5.  Setting up training arguments and a data collator.
6.  Running the fine-tuning process using the Hugging Face `Trainer`.
7.  Saving the fine-tuned model and tokenizer.

## Key Features

* **Model Fine-tuning:** Supervised Fine-Tuning (SFT) of the "Qwen/Qwen3-4B" model.
* **Custom Dataset:** Uses a small, specific dataset for ISO 21448 SOTIF.
* **Qwen Chat Format:** Data is formatted to match the required input structure for Qwen chat models.
* **Efficient Training:**
    * Supports `bfloat16` and `float16` for mixed-precision training.
    * Selects appropriate attention implementation (`flash` or `sdpa`) based on hardware capabilities.
    * Manages GPU memory with configurable `max_memory` and a utility to clear cache.
* **Hugging Face Ecosystem:** Utilizes `transformers.Trainer` for a streamlined training loop.
* **Output:** Saves the trained model and tokenizer for later use.

## After Fine Tuning on RTX3090 for 15min
```
You: scope of ISO 21448:2022
Bot: ISO 21448:2022, titled **"Road vehicles — Safety of the Intended Functionality (SOTIF)"**, is a critical standard that addresses the safety of road vehicles in the context of advanced technologies, 
particularly those involving automated driving systems. It complements ISO 26262, which focuses on functional safety, by addressing safety risks that arise even when systems are designed to be safe but may not perform as intended due to limitations in design, testing, or real-world conditions.

---

### **Scope of ISO 21448:2022**

ISO 21448:2022 applies to **road vehicles**, including **passenger cars, commercial vehicles, and other road vehicles**, and covers **electronic and electrical systems** that support **automated driving 
functions**. These systems may include:

- **Sensors** (e.g., cameras, lidar, radar)
- **Actuators** (e.g., steering, braking, throttle)
- **Control systems** (e.g., adaptive cruise control, lane-keeping assist)
- **Communication systems** (e.g., V2X, vehicle-to-vehicle communication)

The standard is specifically designed to address **safety risks caused by functional insufficiencies**, such as:

- **Inadequate performance** of sensors or algorithms
- **Limitations in system design** (e.g., insufficient redundancy)
- **Errors in system behavior** due to unexpected inputs or environmental conditions
- **Inadequate testing or validation** of system performance under real-world conditions

---

### **Key Objectives of ISO 21448:2022**

1. **Ensure Safety of Intended Functionality (SOTIF):**
   Prevent or mitigate risks caused by functional insufficiencies, even when the system is designed to be safe.

2. **Address System Limitations:**
   Identify and manage risks arising from system limitations, such as:
   - Incomplete sensor coverage
   - Algorithmic biases
   - Inadequate handling of edge cases

3. **Support System Validation and Testing:**
   Provide guidelines for testing systems under diverse and challenging conditions, including:
   - Real-world driving scenarios
   - Edge cases not covered in standard testing
   - System interactions with other road users or infrastructure

4. **Facilitate Risk Management:**
   Help organizations identify, assess, and mitigate risks associated with SOTIF, including ...

```


```
You: Give me example of functional insufficies according to ISO 21448 
Bot: ISO 21448:2021, *Road vehicles — Safety of the Intended Functionality (SOTIF)*, provides a framework for ensuring the safety of road vehicles by addressing **functional insufficiencies**—deficiencies in the intended functionality of a system or vehicle that could lead to safety risks. Functional insufficiencies are categorized into **functional insufficiencies of the system** and **functional insufficiencies of the vehicle**, each with specific types and examples.

Here is an example of a **functional insufficiency** according to ISO 21448, categorized under **Functional Insufficiencies of the System**:

---

### **Functional Insufficiency Type: Incomplete Functional Coverage**

#### **Description:**
The system does not cover all possible scenarios or conditions that could lead to a safety risk, even though it is designed to handle the intended use cases.

#### **Example:**
A **level 2 highway driving assist system** (e.g., adaptive cruise control with lane-keeping assist) may not account for **sudden lane changes by other vehicles** or **unexpected roadworks** that are not detected by the system’s sensors. While the system is intended to maintain lane position and speed, it may fail to detect and respond to these unexpected events, leading to a risk of collision or loss of control.

#### **Safety Risk:**
- **Collision with other vehicles** due to insufficient detection or response.
- **Loss of control** if the system cannot handle unexpected maneuvers or environmental changes.

#### **Mitigation Measures:**
- **Enhanced sensor coverage** (e.g., additional cameras, radar, or lidar).
- **Improved algorithm logic** to detect and respond to edge cases.
- **Driver monitoring systems** to alert the driver in case of system limitations.
- **Clear user guidance** about system limitations and when to take over control.

---

### **Functional Insufficiency Type: Inadequate Functional Performance**

#### **Description:**
The system performs its intended function but with insufficient accuracy, reliability, or robustness, leading to safety risks.

#### **Example:**
A **level 3 autonomous driving system** may have **inaccurate object detection** in low-light conditions or under adverse weather (e.g., heavy rain or fog). While the system is designed to detect and avoid obstacles, its performance may degrade, leading to a failure to detect a pedestrian or vehicle in its path.

#### **Safety Risk:**
- **Collision with pedestrians, cyclists, or...
```
