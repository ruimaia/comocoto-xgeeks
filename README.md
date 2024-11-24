# comocoto-xgeeks

## Description  
This application is designed to help businesses efficiently manage and respond to budget requests. By leveraging **Large Language Models (LLMs)** and **Machine Learning (ML)**, the solution parses historical email interactions into structured data and predicts budgets for new customer requests. The approach ensures timely and informed responses, even for companies with limited resources.  

## Problem Statement  
Small-to-mid-sized companies like **Company A**, a window framing business, often face challenges handling a surge in budget requests after setting up their online presence. Due to limited resources, the company cannot respond to all requests promptly, leading to potential missed opportunities. Manual handling of unstructured text (e.g., emails) is slow and prone to inconsistencies, making it crucial to streamline this process.  

## Proposed Solution  
Our solution combines **LLMs** and **ML** to address this bottleneck:  

1. **Data Parsing with LLMs**:  
   - Historical email interactions are parsed into structured data, extracting key details like specifications and customer requirements.  
   - Synthetic datasets were generated using LLMs to simulate historical and new customer requests for proof-of-concept.  

2. **Budget Prediction with ML**:  
   - A simple **k-Nearest Neighbors (KNN)** regression model was chosen for its explainability, allowing new requests to be matched with the most similar historical ones.  
   - The KNN approach provides transparency by highlighting comparable cases for each prediction, increasing trust in the model’s outputs.  

3. **Streamlined Workflow**:  
   - Automating the initial request handling process saves time and ensures consistent responses.  
   - Businesses can scale their operations without needing significant manual intervention.  

This proof-of-concept showcases how AI can help small-to-mid-sized businesses optimize their workflows, enhance customer satisfaction, and unlock growth opportunities.