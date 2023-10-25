# Detection_Network_Anomalies


This project aims to detect attacks in hospital networks:

The rapid expansion of computer networks and the proliferation of applications running on them have amplified the significance of network security. Security vulnerabilities are pervasive in computer systems, posing challenges that are both technically intricate and financially burdensome for manufacturers to address comprehensively. Consequently, Intrusion Detection Systems (IDSs) are assuming a pivotal role as specialized devices for identifying abnormal activities and attacks within networks.

Traditionally, research in intrusion detection has predominantly concentrated on two techniques: anomaly-based and misuse-based detection. Misuse-based detection, which is preferred in commercial products due to its predictability and high accuracy, has been the focal point of much academic research. However, anomaly detection is deemed a more potent approach due to its theoretical capacity to detect novel attacks.

A thorough examination of recent research in anomaly detection reveals several machine learning methods boasting a remarkable 98% detection rate while maintaining a low false alarm rate of 1%. Surprisingly, the utilization of anomaly detection techniques is conspicuously absent in state-of-the-art IDS solutions and commercial tools. Practitioners often perceive anomaly detection as an immature technology, creating a significant contrast between academic research and real-world applications.

Addressing this paradox requires extensive research into anomaly detection, considering various factors such as learning and detection methods, training and testing datasets, and evaluation methodologies.

The goal is to develop a network intrusion detection system capable of identifying anomalies and attacks within the network. This problem can be divided into two primary challenges:

1. **Binomial Classification:** Determining whether network activity is normal or indicative of an attack.

2. **Multinomial Classification:** Identifying whether network activity is normal, indicative of a Denial of Service (DOS) attack, a Probe, an unauthorized Root-to-Local (R2L) access attempt, or an unauthorized User-to-Root (U2R) access attempt.

