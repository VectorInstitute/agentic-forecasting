# Day 1 — Time Series Forecasting Foundations

**Outline session:** Time series forecasting foundations (data, experiment design, challenges) (10:00–10:20am) — Ethan Jackson. Covers: why forecasting, brief history of methods and use cases, forecasting experiment design, `aieng-forecasting` package overview. Ends with handoff to Behnoosh's conventional methods session.
**Source file:** GMT20260708-135848_Recording.transcript.vtt

---

WEBVTT

1
00:00:00.000 --> 00:00:01.300
Ethan Jackson: Waiting for this one, Jean

2
00:00:02.040 --> 00:00:26.660
Ethan Jackson: All right. Welcome, everyone. My name is Ethan. I hope that this isn't the first time that you've met me before. We've had some really cool discussions and warm ups to this point. I'll go over some of the context just to make sure that we're all on the same page. And then today, we're really excited to give you a more detailed walkthrough of the concepts and reference implementations that we will get to

3
00:00:26.660 --> 00:00:28.589
Ethan Jackson: explore together in this bootcamp.

4
00:00:29.760 --> 00:00:30.660
Ethan Jackson: Okay.

5
00:00:30.840 --> 00:00:34.599
Ethan Jackson: So again, a little bit of a refresher. Why forecasting?

6
00:00:35.100 --> 00:00:39.580
Ethan Jackson: And again, I'm really thinking of this boot camp as a…

7
00:00:39.580 --> 00:01:02.420
Ethan Jackson: an installment in our series of programming on Agentic AI, and forecasting is a very, very kind of unique and rich testbed for us to kind of push the frontier of Agentic AI and consider it from a more specialized lens. So, you know, we will obviously learn about forecasting and apply it to forecasting use cases, but I really do think this is a great opportunity for us to

8
00:01:02.420 --> 00:01:09.139
Ethan Jackson: take a very modern and refreshed look at Agentic AI in, you know, summer 2026.

9
00:01:09.320 --> 00:01:20.679
Ethan Jackson: About forecasting in particular, one thing that makes it really great is that we've got an opportunity to work with unstructured signals. So Agentic AI, obviously, the, the,

10
00:01:20.830 --> 00:01:31.840
Ethan Jackson: what's powering all of this is the large language model, so we can deal with, unstructured language data, but it can go far beyond, just language. We can…

11
00:01:31.840 --> 00:01:52.910
Ethan Jackson: Reason over combinations of things like news, policy, reports, documents, other kinds of data in addition to numbers. But it's not just limited to numerically formatted context the way that we used to do forecasting. And as we'll see in the boot camp, we could still very, very well still should be using numerical methods.

12
00:01:52.910 --> 00:01:57.470
Ethan Jackson: But there's an opportunity to combine these under one kind of new paradigm.

13
00:01:58.100 --> 00:02:08.210
Ethan Jackson: One thing that's really great is from an evaluations perspective. There's an objective ground truth for forecasting problems. Every prediction is scored against what actually happened.

14
00:02:08.280 --> 00:02:32.299
Ethan Jackson: This is a topic that we're going to cover a lot in the over the next days. Evaluation doesn't become trivially easy, especially in backtesting. But the great thing is that if we're kind of thinking about the evaluation of frontier AI agents, forecasting offers an opportunity to have an objective ground truth that's very difficult to kind of

15
00:02:32.300 --> 00:02:35.620
Ethan Jackson: pollute in terms of evaluation faithfulness.

16
00:02:36.200 --> 00:02:49.210
Ethan Jackson: The next thing is that agents can learn from experience. This is one of my favorite, I would say, kind of mindsets of 2026, is it really seems like a lot of the field is

17
00:02:49.630 --> 00:03:09.569
Ethan Jackson: thinking much more deeply about the ways in which agents can accumulate experience. And forecasting is a really, really interesting way to do this, because it's kind of naturally episodic. If you think about it being deployed in a live context, and it's very non-stationary. So it's it's

18
00:03:09.570 --> 00:03:11.570
Ethan Jackson: Provides a very challenging scenario.

19
00:03:11.570 --> 00:03:16.630
Ethan Jackson: context in which to study Agentic learning, but a very rich one.

20
00:03:16.970 --> 00:03:21.640
Ethan Jackson: And as I kind of alluded to, forecasting provides something like

21
00:03:21.640 --> 00:03:39.619
Ethan Jackson: what I would consider to be an unsaturable benchmark. The problem is always shifting. The future generally, genuinely can't be memorized, and I don't think it can be gamed unless AI systems become that much more powerful that they could manipulate outcomes. But I don't think we're there yet.

22
00:03:41.600 --> 00:03:42.540
Ethan Jackson: Okay.

23
00:03:43.570 --> 00:03:50.230
Ethan Jackson: I've shown this plot before, but I just want to show it again. One of the reasons why

24
00:03:50.650 --> 00:03:51.619
Ethan Jackson: We're, you know.

25
00:03:52.190 --> 00:04:10.890
Ethan Jackson: kind of reminding ourselves, and I'm saying I'm reminding myself that this is a very interesting thing to study, even if it's difficult, is because this benchmark from ForecastBench basically showing that without making any particular effort.

26
00:04:11.790 --> 00:04:15.750
Ethan Jackson: LLMs on their own, and the basic agents that can be created with them.

27
00:04:15.750 --> 00:04:39.300
Ethan Jackson: again, without any further interventions, are continuing to improve kind of model generation over model generation. And there's been a steady increase towards the kind of exceeding the best forecasters in the world on a broad range of tasks. If you include market market data, which are still considered to be some of the most difficult forecasting problems.

28
00:04:39.300 --> 00:04:48.430
Ethan Jackson: We are not quite achieving parity with human superforecasters, even with our most powerful agents, but the trend is heading in that direction.

29
00:04:48.430 --> 00:05:05.740
Ethan Jackson: And that's, for me, motivation of, like, why… why we keep doing this, even if it… if it is… can be frustrating sometimes when you don't get, amazing results every time. But the trend overall is going in this direction. And again, a recurring theme that we'll… that we will be discussing.

30
00:05:07.350 --> 00:05:14.820
Ethan Jackson: So the point here that I want to make is that there's all this kind of Excitement around…

31
00:05:15.370 --> 00:05:19.899
Ethan Jackson: LLMs and agents as being great at forecasting, but

32
00:05:20.000 --> 00:05:26.370
Ethan Jackson: these methods do have to earn their place. Being impressive on a benchmark isn't useful on your problem.

33
00:05:26.510 --> 00:05:33.370
Ethan Jackson: The test is beating established methods in your problem domain, honestly, and this comparison between

34
00:05:33.410 --> 00:05:46.550
Ethan Jackson: conventional methods and agentic methods, is… is a big part of what the bootcamp is built around. I would say it's about… covers about half of the major themes, and we want to treat this very honestly.

35
00:05:47.900 --> 00:05:58.019
Ethan Jackson: So the two questions, and kind of two lenses that we'll focus on, the first again is, is that honesty against conventional methods question, can LLMs and agents forecast?

36
00:05:58.450 --> 00:06:04.240
Ethan Jackson: Do they act as effective time series forecasters measured honestly against real baselines?

37
00:06:05.080 --> 00:06:28.389
Ethan Jackson: The second question is, well, how else does Agentic AI apply? How do advances like self adaptation, learning, Agentic evaluation, how do these carry over to forecasting and prediction tasks? We could be asking an agent to do a lot more than just returning a numerical forecast. As we'll see today, we could ask for analysis, we could ask for scenario modeling.

38
00:06:28.420 --> 00:06:30.170
Ethan Jackson: All kinds of things.

39
00:06:30.370 --> 00:06:36.280
Ethan Jackson: and there's a a few other things that we'll we'll talk more about as we see the reference implementations.

40
00:06:39.190 --> 00:06:40.680
Ethan Jackson: And, again.

41
00:06:40.840 --> 00:06:52.430
Ethan Jackson: going a little bit deeper into the things that are maybe appealing about Agentic Forecasting. Number one, they're flexible. We could be dealing with dense time series, very sparse, irregular events.

42
00:06:52.480 --> 00:07:11.140
Ethan Jackson: text, different kinds of prediction questions, these can all share the same pipeline. We don't have to rebuild, very much just because we are changing the nature of the data that underpins a prediction problem. They're scalable in this… in a sense, that

43
00:07:11.140 --> 00:07:33.980
Ethan Jackson: We have one base LLM or many available base LLMs, and these can be configured for different forecasting tasks very, very easily. So we don't have to rebuild models. We can kind of just take what we have available and focus on the configuration layer. And this makes it very, very easy to scale to a large number of problems.

44
00:07:33.980 --> 00:07:37.340
Ethan Jackson: To customize them to different instances of the same problem.

45
00:07:37.340 --> 00:07:56.630
Ethan Jackson: I won't make the argument that LLMs are naturally scalable in terms of supporting large numbers of predictions. We'll talk a little bit more about this later, but there are definitely ways that we can use LLMs and find the right degree of scale to support those larger forecasting problems.

46
00:07:57.350 --> 00:08:20.819
Ethan Jackson: Number three is that they're steerable. LLMs can be steered or prompted to be highly opinionated. They can hold priors and they can weigh evidence according to different strategies. And I think this is kind of an underutilized feature of LLMs in forecasting. The idea that a diversity of perspectives could be a key to

47
00:08:20.820 --> 00:08:26.010
Ethan Jackson: in increasing the overall robustness and calibration of a forecasting system.

48
00:08:26.340 --> 00:08:45.129
Ethan Jackson: Number four, adaptability. As I mentioned, agents increasingly are being treated as entities that can learn with experience, learn from feedback and experience over time. And in the learn days and over the course of the build days, we'll explore this in much greater detail.

49
00:08:45.510 --> 00:08:48.809
Ethan Jackson: And number 5, they're pluggable, so…

50
00:08:48.810 --> 00:09:13.550
Ethan Jackson: LLMs are very flexible in terms of the way that they can be integrated into different software, into different applications. So we can think of prediction agents or forecasting agents as being highly interoperable across decision pipelines. Depending on what kind of prediction or what kind of information you need from a prediction agent, you can customize what the outputs are so that they can feed into

51
00:09:13.670 --> 00:09:15.959
Ethan Jackson: different decision processes.

52
00:09:19.130 --> 00:09:30.570
Ethan Jackson: So again, we'll look at two major ways over the course of these days to use a forecaster. Number one is as a time series predictor.

53
00:09:30.670 --> 00:09:44.719
Ethan Jackson: So the idea is to emit a standard prediction. These can be scored head-to-head against predictions from conventional methods. We can select the appropriate metric and, kind of run this in a… in different kinds of experiments.

54
00:09:44.720 --> 00:10:08.509
Ethan Jackson: The second track is to really treat these agents as agents, where we can assign them different tasks, ask questions, we can ask them to do exploration, we can ask them to do assessment, analysis, different kinds of things. So they're not necessarily outputting a numerical forecast that can be compared head to head, but still very interesting to look at nonetheless.

55
00:10:08.510 --> 00:10:14.969
Ethan Jackson: And we can bring in evaluation techniques that are better suited to more open-ended agentic tasks.

56
00:10:15.100 --> 00:10:23.470
Ethan Jackson: The interesting thing is that the same exact agent can be used for either one, and we'll see this as a recurring theme.

57
00:10:24.750 --> 00:10:40.639
Ethan Jackson: A little more about the reference implementations that we'll get to know. There's, again, a very basic one for getting started. We have one centered around S&P 500, market price predictions. Banyush will talk more about this this morning.

58
00:10:40.640 --> 00:11:02.479
Ethan Jackson: We have food price, consumer price index prediction that will be paired with a demo on the LLMP or LLM process. I will focus a couple of the presentations on oil price forecasting measured a couple of different ways, and we have a few different lenses of Agentic AI to present in the context of that.

59
00:11:02.480 --> 00:11:03.680
Ethan Jackson: Use case for that.

60
00:11:03.950 --> 00:11:22.880
Ethan Jackson: And then we also have a different kind of prediction problem where rather than looking at numerical time series, we look at a kind of a three class kind of prediction classification problem with Bank of Canada interest rate decision calls. And these open up different ways that we can.

61
00:11:23.450 --> 00:11:25.919
Ethan Jackson: explore the Agentic techniques.

62
00:11:28.040 --> 00:11:38.549
Ethan Jackson: A couple of notes to get the most out of the bootcamp, and this is maybe more geared towards the build days and the time leading up to them.

63
00:11:38.550 --> 00:11:52.260
Ethan Jackson: Please keep in mind that we have a limited compute budget. I strongly recommend that people keep the default set to Gemini Flash Lite, especially if you're running back tests, against conventional methods.

64
00:11:52.260 --> 00:12:02.349
Ethan Jackson: A full backtest on even Gemini 3.5 Flash could burn your entire budget. You'd have to run it a few times on a large task, but it could happen.

65
00:12:02.350 --> 00:12:21.890
Ethan Jackson: What I'm suggesting is that if you get to the point where what you really want to know is whether the more powerful model backtests better than the less powerful model, save that for the end, and then you can kind of consume the rest of your budget on that. But I do think there are more interesting things that we can do to

66
00:12:21.890 --> 00:12:27.850
Ethan Jackson: you know, learn about these agents than running long back tests for for reasons that we'll get into.

67
00:12:28.580 --> 00:12:37.409
Ethan Jackson: We'll focus on breadth, not depth, for these Learn Days. What I… what I'm really hoping is that we all

68
00:12:37.410 --> 00:12:57.829
Ethan Jackson: develop a mental map of how the reference implementations and the code base best connect to your use case, not necessarily to understand every line of code in the repository that we're providing. It's just not possible to cover that in such a short amount of time.

69
00:12:57.940 --> 00:13:16.579
Ethan Jackson: try to stay at a higher level, and then understand how the pieces fit together. And then again, we have introduced this, very lightweight concierge agent, as a means for you to just get started and ask some basic questions to the repository, if you need a little bit of help doing self-directed exploration.

70
00:13:18.210 --> 00:13:34.060
Ethan Jackson: So now, I'm going to move on to a little bit more about time series forecasting as a problem space. This will be at a very high level before I pass it to Benush later this morning to go into more detail about the methods and problems that are supported.

71
00:13:34.270 --> 00:13:36.360
Ethan Jackson: So this will just be a quick overview.

72
00:13:36.550 --> 00:13:48.690
Ethan Jackson: We'll talk about methods, just a short lineage that goes from conventional to LMs and agents, talk about evaluation, and we'll talk about what some forecasts look like, in our,

73
00:13:48.970 --> 00:13:50.070
Ethan Jackson: in our setup.

74
00:13:51.260 --> 00:14:02.959
Ethan Jackson: So this is an extremely short history of methods. We don't have enough time to give a comprehensive overview of conventional statistical machine learning methods.

75
00:14:02.960 --> 00:14:15.669
Ethan Jackson: So just to say, what the ones that are supported in our bootcamp off the shelf are shown in pink here. We have support for statistical, machine learning, and LLM-based methods.

76
00:14:15.890 --> 00:14:40.279
Ethan Jackson: A lot of this is because we're using the Darts library as an aggregator of a whole bunch of conventional methods. They do support deep learning, and they do even support time series foundation models, but our compute environment is not geared towards these. So about the most heavyweight machine learning method we can reasonably support is on the order of XGBoost and LightGBM.

77
00:14:40.580 --> 00:14:44.450
Ethan Jackson: So… So we'll have to make do with that.

78
00:14:44.580 --> 00:15:08.399
Ethan Jackson: Just to give a bit of background, so the statistical methods like ARIMA and ETS are kind of purpose-built statistical methods for time series. They have a very small number of parameters and can be optimized using kind of linear statistical optimizers. So these are super, super fast and they're surprisingly still very robust. Darts provides really good

79
00:15:08.400 --> 00:15:17.139
Ethan Jackson: wrappers around these methods. For example, there's an AutoARIMA that does kind of its own hyperparameter optimization over ARIMA, so you can have

80
00:15:17.390 --> 00:15:35.770
Ethan Jackson: good confidence that you kind of press a button and get a pretty decently calibrated univariate forecast with confidence intervals that are based on repeated Monte Carlo sampling. That's pretty representative of what this class of statistical methods can do.

81
00:15:35.920 --> 00:15:53.449
Ethan Jackson: There are some statistical methods, like vector autoregression, that can add support for covariates, but they're quite finicky to work with. You know, I'm sure we've all worked with linear model optimization. If you've got tons of covariates and they're collinear, that's not going to help you.

82
00:15:53.500 --> 00:16:03.680
Ethan Jackson: So they work best on univariate time series, maybe with a small number of covariates.

83
00:16:03.770 --> 00:16:06.010
Ethan Jackson: but they give you pretty strong baselines.

84
00:16:06.010 --> 00:16:25.550
Ethan Jackson: What I found in my own experiments and research that if you skip the gap all the way to gradient boosted methods, these provide a very, very good way to start incorporating covariates and even relatively large numbers of covariates into numerical forecasting problems.

85
00:16:25.550 --> 00:16:28.020
Ethan Jackson: And these are a great way to do it.

86
00:16:28.300 --> 00:16:45.349
Ethan Jackson: In terms of deep learning and foundation models, what they each added, I would say is, is, like, deep learning really introduced representation learning to time series forecasting, and there's a lot of, interesting stuff that I wish we had time to talk about, but I'm gonna, hold myself

87
00:16:45.380 --> 00:16:59.509
Ethan Jackson: To do it, if you're interested, maybe we can have a side chat. And of course, with time series foundation models, the whole idea is to bring the paradigm of zero-shot transfer that we've seen with large language models and transformers into the time series domain.

88
00:17:00.920 --> 00:17:11.040
Ethan Jackson: But we're going to focus on stats, machine learning, LLMs, and the whole benefit here is that we get language and reasoning from Frontier LLMs.

89
00:17:12.660 --> 00:17:13.609
Ethan Jackson: So…

90
00:17:13.880 --> 00:17:34.129
Ethan Jackson: What unifies all of this together, though, is that we've built a unified prediction interface around all of these methods, at least for time series forecasting problems. So we have built this framework so that you can run statistical methods head to head against Gemini LLM-powered agents.

91
00:17:34.190 --> 00:17:37.019
Ethan Jackson: measured consistently exactly the same way.

92
00:17:38.770 --> 00:17:45.929
Ethan Jackson: Okay, so let's look at how… talk a little bit about how we're going to measure a forecaster, and what makes comparison honest.

93
00:17:47.680 --> 00:18:00.399
Ethan Jackson: The anatomy of a forecast evaluation in our framework is as follows. First, we define a task. This is what to forecast, what our target series is, what our prediction horizon is, and what the frequency of data looks like.

94
00:18:00.650 --> 00:18:13.670
Ethan Jackson: We specify an origin, kind of the date of the forecast, and a cutoff, which is where you stand and where, the… controlling the data that is made available to the model, up to that point.

95
00:18:14.350 --> 00:18:23.789
Ethan Jackson: Then we define a predictor, which is how we want to answer the forecasting question. This part does vary, but the point is that we can,

96
00:18:24.320 --> 00:18:36.789
Ethan Jackson: shape the, what we want, the prediction, like, the format of the prediction, depending on which, on what the prediction problem is, or what the task is, and then, kind of adhere

97
00:18:36.790 --> 00:18:45.719
Ethan Jackson: each of the methods, or configure each of the methods that, that we want to experiment with so that they produce predictions that are compliant, with this interface.

98
00:18:46.070 --> 00:19:02.250
Ethan Jackson: Then we have a resolution, which is the ground truth at the target date. The resolution is related to the horizon. It depends on whether, you know, we're predicting one day ahead, or maybe one year ahead, as is in the case of the food price prediction implementation.

99
00:19:02.460 --> 00:19:09.850
Ethan Jackson: And once we have resolutions available, then we can compute scores. How close?

100
00:19:09.850 --> 00:19:26.109
Ethan Jackson: was the, the probabilistic answer was. In this bootcamp, we're focusing, almost entirely on probabilistic scoring, so every method that we've, that we've built support around produces a, a probability distribution with

101
00:19:26.110 --> 00:19:31.880
Ethan Jackson: More or less fixed quantiles, so that we can have a,

102
00:19:31.880 --> 00:19:50.500
Ethan Jackson: consistent way of representing probability distributions around forecasts, and the metrics that we have built around are the ranked probability score or the continuous variant of that, or the Brier score in the case of binary predictions. We'll talk more about each of these later.

103
00:19:51.990 --> 00:19:54.310
Ethan Jackson: And so, what makes a comparison honest?

104
00:19:54.460 --> 00:20:10.150
Ethan Jackson: this will be a recurring theme. For numerical methods, this is easy. We fit the model so that data is… the data that a model sees is cut off at the origin, or with a cutoff date if we want to provide it.

105
00:20:10.540 --> 00:20:23.459
Ethan Jackson: So these are cutoffs safe by construction. We just have to handle the data correctly, and make sure that we're enforcing the cutoffs. That makes it really easy to backtest any period if you have data for it.

106
00:20:23.750 --> 00:20:41.290
Ethan Jackson: With LLMs, it's trickier. The Gemini class of models that we're using, the stated knowledge cutoff is up to January 2025. But it's unclear to me whether updates to those models could potentially leak in information from beyond those cutoff dates.

107
00:20:41.510 --> 00:20:43.720
Ethan Jackson: Before the cutoff.

108
00:20:43.850 --> 00:20:55.549
Ethan Jackson: You could argue that information is being recalled because, depending on the prediction task, the LLM has maybe seen the literal ground truth, or at least has seen the context surrounding the ground truth.

109
00:20:55.590 --> 00:21:06.589
Ethan Jackson: And we could argue that these are honest only after the cutoff, but I'm gonna go further and say that it's, like, there… this… anything that we get out of an LLM,

110
00:21:06.680 --> 00:21:16.520
Ethan Jackson: at all is an optimistic forecast. It makes it very, very difficult to get a perfectly honest, evaluation.

111
00:21:16.850 --> 00:21:23.609
Ethan Jackson: So we do our best, and we score LLM on protected post-cutoff windows, which does…

112
00:21:23.780 --> 00:21:30.269
Ethan Jackson: really force us to work with more recent data as recent as possible.

113
00:21:32.730 --> 00:21:41.520
Ethan Jackson: In the bootcamp, we make a distinction between backtest and protected eval, just as a way of…

114
00:21:41.920 --> 00:21:42.739
Ethan Jackson: like,

115
00:21:42.880 --> 00:21:52.740
Ethan Jackson: having some kind of of named separation between what you think of as a back testing set and what you think of as an evaluation set.

116
00:21:52.820 --> 00:22:04.920
Ethan Jackson: The point is that for backtesting, you might allow yourself to do many iterations of experiments, as if you were doing hyperparameter optimization, but then you still want to test

117
00:22:05.020 --> 00:22:13.190
Ethan Jackson: Your best selected model, kind of out of sample, on a withheld data set, and you want that to be the most recent data.

118
00:22:13.380 --> 00:22:22.890
Ethan Jackson: How much of the data that you want to use to do that out-of-sample evaluation is always, like, a tension in forecasting, because you want to have enough

119
00:22:22.890 --> 00:22:37.250
Ethan Jackson: data, to gain confidence in, but the further back you go, the kind of less recency bias you have in that… in that answer. So there's… there's always a tension here. And then with LLMs, we kind of have this natural

120
00:22:37.250 --> 00:22:44.220
Ethan Jackson: forcing function, which is the, the, the knowledge cutoff date. So this is just something to be, to be aware of.

121
00:22:47.690 --> 00:23:11.040
Ethan Jackson: I'll let Bhinush talk a little bit more about why we're using the CRPS, the continuous rank probability score. But I will just say briefly that the point is that if you look at, you know, if we imagine that you have different forecasts where one is a very wide probability distribution around a point forecast, one is a very narrow one.

122
00:23:11.040 --> 00:23:20.470
Ethan Jackson: We would… we would prefer to select for forecasts that are both accurate and narrow, and the CRPS gives us a way of…

123
00:23:20.470 --> 00:23:24.650
Ethan Jackson: Measuring this, even if the point forecasts were exactly the same.

124
00:23:27.210 --> 00:23:38.370
Ethan Jackson: Just a little sneak peek into the code. Again, the point is that when we start to work with predictors, these are interfaces in our code.

125
00:23:38.370 --> 00:23:48.299
Ethan Jackson: where you have a forecasting task and a context that carries some information about the… the task with it. This is stuff like the… the, data series and…

126
00:23:49.340 --> 00:23:54.869
Ethan Jackson: Like, descriptions of, of, of, of the time series that are involved, etc.

127
00:23:55.040 --> 00:24:10.359
Ethan Jackson: But basically, anything that implements this interface can be incorporated into an experiment, and we've implemented quite a few that, again, span from naive and statistical baselines through to LLMs and agents.

128
00:24:10.970 --> 00:24:12.569
Ethan Jackson: All with one interface, right?

129
00:24:14.160 --> 00:24:31.409
Ethan Jackson: Starting to look at a couple of forecasts, if you… if you look at… and this is one of the ones included in the… in the Getting Started, if you look at an ARIMA forecast for market price data, or even, like, energy price data here.

130
00:24:31.410 --> 00:24:40.869
Ethan Jackson: You'll… you'll quickly notice that these kind of extrapolation-based forecasts are… are not robust to regime change.

131
00:24:41.060 --> 00:24:42.110
Ethan Jackson: And…

132
00:24:42.450 --> 00:25:00.720
Ethan Jackson: even here, you know, you don't… you can just look at it and see it's just, like, it looks like an out-of-phase persistence forecast, as if it was just kind of forecasting a slight variation of the last known value, and the confidence intervals are also not, like, reacting to anything. The…

133
00:25:00.790 --> 00:25:10.229
Ethan Jackson: the… you can really see here that there's nothing special going on here. There's no calibration with respect to,

134
00:25:11.040 --> 00:25:13.449
Ethan Jackson: Like, context, let's say.

135
00:25:13.830 --> 00:25:26.870
Ethan Jackson: So, essentially, we just see that there's an extrapolation. If you look here, this is a measure of errors on forecasts for gasoline prices over, like, the last 20, 25 years.

136
00:25:26.870 --> 00:25:35.540
Ethan Jackson: And the biggest misses are exactly where you might expect them to be, knowing what's happened over this time frame.

137
00:25:35.540 --> 00:25:38.100
Ethan Jackson: But the point is, is that like, you know.

138
00:25:38.400 --> 00:25:55.140
Ethan Jackson: there's nothing about this time series that gave it any context that these spikes were coming, and the hope is that an Agentic forecaster could learn to pay attention well enough to what's going on in the world that maybe it can't predict the magnitude of spikes, but

139
00:25:55.290 --> 00:26:07.749
Ethan Jackson: could potentially predict that things like volatility are increasing or uncertainty is increasing. And even widening the confidence intervals around the periods of these spikes could be something useful.

140
00:26:07.920 --> 00:26:10.219
Ethan Jackson: But already we can see at least

141
00:26:10.640 --> 00:26:25.479
Ethan Jackson: it's not a terrible picture. The extrapolation forecast, that we got out of this were, you know, you get one error metric of 10 CRPS, and autoerema is still doing better than that. So, it's not…

142
00:26:25.850 --> 00:26:28.990
Ethan Jackson: For nothing, but I think we can do a lot better.

143
00:26:30.650 --> 00:26:45.260
Ethan Jackson: What to take forward from this point? Again, recurring theme, honest evaluation always. We have an interface that enforces strict cutoffs and protected evals for every method.

144
00:26:45.440 --> 00:26:53.680
Ethan Jackson: But these are optimistic when using LLMs and agents for the reasons that, again, we'll discuss more.

145
00:26:54.310 --> 00:27:14.229
Ethan Jackson: Established methods set a very good baseline bar. In some of the cases that we'll see today, the baseline methods are still outperforming the agents that we've built. We'll say that we haven't fully experimented and calibrated with these, but you should have the expectation that these are not going to be easy to beat, especially if you spend effort on them.

146
00:27:14.460 --> 00:27:24.510
Ethan Jackson: Good classical model clearly beats, the naive floor in this case, and these are tough bars to beat, especially for these difficult

147
00:27:24.670 --> 00:27:26.740
Ethan Jackson: market prediction problems.

148
00:27:27.970 --> 00:27:46.249
Ethan Jackson: But we're optimistic that agents can earn their place. Again, even in something like market prediction problems, bringing in context and reasoning at the shocks, this is something that we want to test, and we don't want to assume that this is going to work.

149
00:27:47.810 --> 00:27:56.939
Ethan Jackson: So I think that's going to conclude my section for now. And I will pass it over to Benush to continue with the reference implementations.
