# High level schedule (tentative)

## Day 1 — Morning (9:30 am – 12:00 pm)

| Time | Session | Duration |
| :---- | :---- | :---- |
| 9:30 – 10:00 am | Greeting, admin, intro (Many, incl. Ethan) | 30 min |
| 10:00 – 10:20 am | Time series forecasting foundations (data, experiment design, challenges) (Ethan) | 20 min |
| 10:20 – 10:50 am | Overview of conventional methods // financial markets reference implementation (Behnoosh) | 30 min |
| 10:50 – 11:00 am | *Break* | 10 min |
| 11:00 – 11:30 am | Overview of LLM processes // Canada's Food Price Report reference implementation (Ali) | 30 min |
| 11:30 am – 12:00 pm | Presentation of "Analyst Agent" (reference code exec agent w/ agent skills) // energy markets reference implementation (part 1\) (Ethan) | 30 min |
| 12:00 – 1:00 pm | *Lunch* | 60 min |
| 1:00 – 3:00 pm | Afternoon facilitation session | 120 min |

## Day 2 — Morning (9:30 am – 12:00 pm)

| Time | Session | Duration |
| :---- | :---- | :---- |
| 9:30 – 10:00 am | Overview of agentic AI evaluation applied to forecasting // Bank of Canada rate decision reference implementation (Ali) | 30 min |
| 10:00 – 10:30 am | Presentation of "Adaptive Agent" (reference agent that uses special skills to update its own forecasting strategy) // energy markets reference implementation (part 2\) (Ethan) | 30 min |
| 10:30 – 10:35 am | *Break* | 5 min |
| 10:35 – 10:55 am | Self-improving agentic systems (micro-lecture: Automated Design of Agentic Systems, Darwin Gödel Machine) (Ethan) | 20 min |
| 10:55 – 11:00 am | *Break* | 5 min |
| 11:00 am – 12:00 pm | Industry spotlight lecture — Matin Yousefabadi, Unilever Horizon3 Labs | 60 min |
| 12:00 – 1:00 pm | *Lunch* | 60 min |
| 1:00 – 3:00 pm | Afternoon facilitation session | 120 min |

# Detailed schedule

Day 1 opening – signal that the learn days will be organized around two themes. On each day, we will spend time on both concepts and implementations. 

Frame the learn days around two key questions:

1. Can LLMs and agents act as effective time series forecasters?  
2. How can further advances in agentic AI apply to forecasting?

### Day 1 Content

- Intro to time series forecasting (Ethan)  
  - Brief history of methods and use cases  
  - Forecasting experiment designs  
  - `aieng-forecasting` package overview  
- Conventional methods // SP500 Market Price Forecasting  
  - Univariate vs. multivariate  
  - Classes of methods: stats, ML, deep learning, time series foundation models  
  - Key challenges  
    - Data  
      - Dataset size  
      - Regularity / sparsity  
      - Feature engineering / maintaining context as covariates  
    - Struggle with context shift / not adaptable to new sources of information  
  - First prediction task  
    - Can concretely introduce backtest / eval configs that specify an experiment  
    - Show how multiple predictors implement the Predict interface and can “compete” head to head in an experiment.  
    - Basically walk through a notebook, focusing more on concepts than code.  
- LLM Processes  
  - Conceptual overview, drawing on seminal paper from Requima/**Duvenaud**, also “Context is Key” paper. Answers: What are LLM Processes and how well do they work? In what scenarios do they work well?  
  - Show (still in slides) how our implementation of LLM Processes is designed. Show system prompt excerpts, how data are packed for input, and show how there are different kinds of output formats.  
  - Discuss backtesting limitations with LLM training cutoff and possible knowledge updates (some models show 2 different cutoff dates – dig into this)  
  - Then we can migrate to the code, where we can walk through one of the food CPI notebooks that focuses specifically on the time series forecasting problem, which includes the use of historical Canada’s Food Price Report (CFPR) documents as context.   
  - This will give us a chance to introduce the code for PDF extraction, too.   
  - We can simultaneously use this as an intro to the reference implementation for CFPR from both a technical/code and experimental/scientific perspective.  
  - On Day 1 we specifically do not cover the notebook that sets up an Evaluation pipeline in which the LLM-as-a-judge uses the CFPR reports to assess the alignment between agentic reasoning and human reasoning. (We will save that for Day 2.)  
- Analyst Agent  
  - Theme: We can configure agents as a natural extension of LLMP  
  - Overview of our basic agentic solution and its components  
  - Show how it can be configured to act as a predictor  
  - Then we can demo it in the context of the Energy prices use case  
    - News agent including the data leakage tension for agents w/ live data access  
    - Agent skills and code execution tool for analysis  
    - Discuss whatever strategy we are actually using for that agent (system prompt, skill configs, strategy guidelines if applicable)  
- Day 1 Closing – Today was all about understanding where off-the-shelf LLMs and AI agents fit into the pantheon of time series forecasting methods. Tomorrow we will explore how other concepts in agentic AI like self-adaptation and agentic evaluations apply to forecasting. We will also have an industry spotlight talk from our friend Matin at Unilever’s Horizon 3 Lab. 

### Day 2 Content

#### Agentic AI Evaluation & BoC Interest Rate Decision Prediction

- Introduce BoC rate prediction problem as a discrete event prediction, e.g. P(cut) at next meeting.  
  - Doesn’t need to focus on a detailed experiment, the focus is more like: This is what the agent’s prediction is. Let’s focus more on the *why.*  
  - We can get to the point of looking at a trace and then ask – how does this agent’s reasoning process compare to the BoC’s official report for that period?  
- Recall basic concepts of agentic AI evaluation. Traces & metrics. Quantitative and qualitative evaluations, LLM-as-a-judge paradigm.  
- Introduce the idea of using the LLM-as-a-judge for *reasoning alignment evaluation.*  
- Show how this works in the reference implementation.

#### Adaptive Agent & Energy use case part 2

- Introduce the Adaptive Agent   
  - Uses a strategy  
  - Has skills to update the strategy, which is also implemented as a skill.   
  - Inspired by techniques like Automated Design of Agentic Systems (ADAS) and Darwin Gödel Machine  
  - We’ll talk about these concepts more later in the morning.   
- Show the concrete implementation of the Adaptive Agent in the Energy use case.   
  - Note that it is just another configuration of our base agent – we’ve just defined a couple of extra pieces to give it extra capabilities.  
  - Show some experimental results. 

#### Self-Improving Agentic Systems

- Take some time to talk about ADAS and DGM and our Adaptive Agent as part of a broader subfield of agentic AI.   
- While the techniques focus on searches for agentic programs (prompts, code) the same techniques can be applied to general AI-driven search, which includes recursive self-improvement (RSI) research.

#### Industry Spotlight Lecture w/ Matin

- Content to be finalized.