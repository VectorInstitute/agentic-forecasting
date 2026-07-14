# Day 2 — Agentic AI Evaluation & Bank of Canada Rate Decision Prediction

**Outline session:** Overview of agentic AI evaluation applied to forecasting // Bank of Canada rate decision reference implementation (9:30–10:00am) — Ali Kore
**Source file:** GMT20260709-133607_Recording.cutfile.20260709194145223.transcript.vtt (part 1 of 2 — this recording covers two back-to-back topics)

---

WEBVTT

1
00:00:02.220 --> 00:00:09.089
Ali Kore: Good morning. Welcome to day 2 of our learn days. We're going to kick things off with

2
00:00:10.060 --> 00:00:16.589
Ali Kore: sort of today's question, which It's particularly useful when you're building agents, so…

3
00:00:16.960 --> 00:00:24.279
Ali Kore: When an agent makes a prediction, we kind of ask the question, how do we know? How do we actually know if it's a good prediction?

4
00:00:24.590 --> 00:00:29.740
Ali Kore: And we'll argue throughout the presentation that a correct answer usually isn't enough, and it's…

5
00:00:29.890 --> 00:00:32.890
Ali Kore: useful as well as important to evaluate the reasoning behind it.

6
00:00:33.060 --> 00:00:40.390
Ali Kore: And throughout the set of slides, our running example is going to be the prediction task on the Bank of Canada rate decision.

7
00:00:42.200 --> 00:00:51.109
Ali Kore: So more specifically, the three points of discussion. First, we're going to go over the prediction task itself. So the Bank of Canada decision as a discrete event.

8
00:00:51.250 --> 00:00:57.330
Ali Kore: And then second, we're going to score the answer where we get kind of an honest result against other baselines.

9
00:00:57.700 --> 00:01:07.040
Ali Kore: And third, which is kind of like the real point of the talk, is we look at judging the reasoning of the agent rather than just the answer. We look at kind of the methodologies that help us do that.

10
00:01:09.200 --> 00:01:22.449
Ali Kore: So looking at the task, so eight times a year, the Bank of Canada announces a rate decision, and the task basically is to predict the direction. So it's three possible outcomes, which is cut, hold, or hike.

11
00:01:23.370 --> 00:01:26.179
Ali Kore: And you can model that as a probability distribution.

12
00:01:26.540 --> 00:01:33.670
Ali Kore: So there are 2 design choices here that matter. You could consider this to be kind of an ordered outcome where

13
00:01:34.000 --> 00:01:45.599
Ali Kore: being wrong where the model goes towards hike when it's actually cut is considerably worse than saying hold. So you can imagine kind of like a spectrum where you have hike on one side and then hold in the middle, and then sort of a

14
00:01:45.790 --> 00:01:47.509
Ali Kore: A cut on the other end.

15
00:01:47.890 --> 00:01:48.910
Ali Kore: Oh.

16
00:01:49.240 --> 00:02:04.720
Ali Kore: In terms of the lead time we give, we forecast 28 days out, particularly on purpose, where we take the day before the two-year bond yield has already priced — before the two-year bond yield has already priced the decision in. So avoiding that bit of target leakage creates a forecast.

17
00:02:04.720 --> 00:02:09.379
Ali Kore: That doesn't just sort of like read the market already pricing the decision in.

18
00:02:12.010 --> 00:02:24.710
Ali Kore: And before we kind of get into scoring anything, we're going to look at what the agent produces. So getting into the idea of traces for agents. So this is an example of a real trace from January 2025 from the agent prediction.

19
00:02:25.040 --> 00:02:29.259
Ali Kore: It emits a distribution here where it says 85% on cut.

20
00:02:29.400 --> 00:02:32.300
Ali Kore: plus a rationale and the signals that it's cited.

21
00:02:33.040 --> 00:02:45.330
Ali Kore: It cites things like an easing, an easing cycle, things like inflation being near target, rising unemployment, things like that. And you can see that everything is traced in LangFuse, so it's possible to inspect every step that the agent takes on.

22
00:02:45.550 --> 00:02:54.179
Ali Kore: And holding on to kind of this, rationale that the agent produces, it's kind of gonna be the thing that we're gonna judge later, as we sort of look at the reasoning that

23
00:02:54.450 --> 00:03:06.540
Ali Kore: So in this case, when you look at the agent's choice, the bank did cut in January, so it's, correct, but when we look deeper into, like, the rationale the agent uses for why it chose cut, we're gonna kind of see that it's not as simple as, it

24
00:03:07.630 --> 00:03:13.840
Ali Kore: So we'll start with the natural kind of conventional thing when we look at basic metrics that help to score the answer directly.

25
00:03:14.540 --> 00:03:28.570
Ali Kore: So the first thing we need in this case is a metric. And for the most part, yesterday, everything we covered was continuous. And I mentioned that we scored it, used a pretty useful metric CRPS for those continuous scores.

26
00:03:28.880 --> 00:03:34.600
Ali Kore: But in this case the rate decision is discrete. So we kind of moved to its discrete variance.

27
00:03:34.930 --> 00:03:41.769
Ali Kore: So when you consider a binary question like cut or not, you could leverage something like the beer score, which is just the mean squared error on the probability.

28
00:03:42.170 --> 00:03:52.159
Ali Kore: But for something like, our particular task, which is a three-way ordered outcome, you would leverage something fairly well-suited for it, which is the RPS, the ranked probability score.

29
00:03:52.350 --> 00:03:57.590
Ali Kore: And the useful thing about this is that it's distance-aware, so it punishes the model for putting mass on hike

30
00:03:57.650 --> 00:04:12.569
Ali Kore: when they cut more than putting it on hold. So it sort of rewards the model of being more conservative about things where it's uncertain. So estimating uncertainty is sort of also, I would say, scored within the metric itself.

31
00:04:13.500 --> 00:04:26.860
Ali Kore: So, the cool thing about RPS is that it basically reduces to Breer score, which is two categories. So, you can imagine that CRPS, Breer, and RPS are just one family of metrics, and they all represent a scenario where, like, a lower metric score is better.

32
00:04:28.580 --> 00:04:34.109
Ali Kore: And now, if you look at the numbers on kind of a protected post cutoff window.

33
00:04:34.430 --> 00:04:44.909
Ali Kore: The bar to beat kind of the baseline is kind of a climatology aspect. So we have a scenario where we always predict the base rates around 76% tends to hold.

34
00:04:45.250 --> 00:04:50.419
Ali Kore: And then you could think about other methods like a conventional logistic regression, which

35
00:04:50.540 --> 00:04:53.259
Ali Kore: Tend to beat it by around 38%.

36
00:04:53.520 --> 00:04:59.430
Ali Kore: And then the agent beats the two by about 29%. But the interesting thing is that the agent loses to the logistic model.

37
00:04:59.680 --> 00:05:01.240
Ali Kore: So,

38
00:05:01.450 --> 00:05:08.699
Ali Kore: That's kind of a good baseline to set here, but putting it more into context in terms of why the agent trails.

39
00:05:09.290 --> 00:05:22.589
Ali Kore: we sort of establish here that this is kind of like the agent's default configuration, so it sees exactly the same four macro features as the logistic model, and nothing more, so there's no aspects of what you would expect with an Agentic model where

40
00:05:22.590 --> 00:05:31.959
Ali Kore: Or, sorry, an Agentic Harness, where you have things like web search, or code execution, or extra tools beyond that. In this case, it's just a single stateless call, so a single analyst passed.

41
00:05:32.120 --> 00:05:40.529
Ali Kore: that's fixed from the moment that we define it. It doesn't particularly study the data or carry kind of a learned strategy from meeting to meeting as it goes over the time series.

42
00:05:41.060 --> 00:05:46.419
Ali Kore: So, roughly speaking, we didn't really tune it much here, and that's kind of deliberate.

43
00:05:46.420 --> 00:06:04.689
Ali Kore: The heavier agent optimization work, as well as the adaptive and self-improving agent work that Ethan is going to get into in later presentations, was built for the energy use case, where that daily oil data gives you a much faster feedback loop to iterate on for the configuration. So it's best to read this as kind of a floor where you have an untuned agent.

44
00:06:04.740 --> 00:06:15.239
Ali Kore: that clears climatology by a decent gap, and then we would sort of close that gap to the other conventional methods and sort of the more complex methods that Ethan covers.

45
00:06:17.480 --> 00:06:32.520
Ali Kore: So coming back to the score itself, a single aggregate of score tends to hide a lot. So this is a plot of the agent's probability of a cut at each meeting against what the bank actually did. So green on this plot is a cut and orange is a hold.

46
00:06:33.120 --> 00:06:43.429
Ali Kore: And if you look at mid-2025, agents kind of sat at 65% to 85% from the cut, meeting after meeting, while the bank held. And eventually, it kind of catches up and begins to align, but…

47
00:06:43.430 --> 00:06:57.720
Ali Kore: if you look at kind of the aggregate RPS here, it's fine, but you can kind of see there's a clear directional bias here, and it's the kind of thing you look at when you really look underneath the hood of what the model's doing, and which is a good segue into kind of the real question of the talk, which is that

48
00:06:57.850 --> 00:07:15.969
Ali Kore: A good score isn't always the same as a good forecast. So on a 3 way decision with a with a particularly dominant class, you can be right a lot by just kind of leaning on the base rate. and you can be wrong in a meeting where your reasoning was actually excellent as well. So if we only track Rps, we can't really tell these 2 apart.

49
00:07:16.200 --> 00:07:25.730
Ali Kore: And particularly going into sort of like a self-improving loop for agents. If we want to create an agent that we trust, but also an agent that can improve, we need to be able to evaluate the reasoning itself.

50
00:07:27.190 --> 00:07:33.880
Ali Kore: So that brings us into the bulk of the talk, where we explore how we evaluate the reasoning to probe whether the model's right for the right reasons.

51
00:07:34.590 --> 00:07:46.239
Ali Kore: So before we get into that, just a quick recap of Agentic evaluation. Every agent run is a trace, which is a kind of a tree of spans, and spans are effectively like tool calls and model calls that we can attach particular scores to.

52
00:07:46.750 --> 00:07:50.739
Ali Kore: And there are 2 complementary ways to

53
00:07:50.780 --> 00:08:04.209
Ali Kore: score these traces. Particularly one is quantitative. It's what we just went into in the past few slides with the, the Breer scores and, like, the CRPS scores and RPS scores, where we just directly judge the label.

54
00:08:04.240 --> 00:08:13.419
Ali Kore: And then qualitative, which we're going to get into when we judge the reasoning, where you use a strong LLM as a judge for things that can't be reduced to a single number, like whether the reasoning is sound.

55
00:08:13.490 --> 00:08:23.220
Ali Kore: And both of these can just attach a scores on the same length. Use trace when you just especially if you just elicit the the Lm. Is judged to produce its reasoning as a metric.

56
00:08:23.740 --> 00:08:31.830
Ali Kore: and we'll get kind of a a bird's eye view of that Lm. As a judge as well. So for each meeting we could take the each's rationale and the signals that are cited.

57
00:08:31.990 --> 00:08:39.230
Ali Kore: And we give a strong model, the bank's own press release for that decision, which has its own kind of rush, almost like the reasoning trace for the bank itself as a ground truth.

58
00:08:39.550 --> 00:08:52.989
Ali Kore: And the judge can return an alignment score from 0 to 1, considering the specific signals that generally overlap and then providing a justification as well for its reasoning on why those two rationales align as well.

59
00:08:53.500 --> 00:09:02.189
Ali Kore: So the key instruction here is to judge the reasoning, not necessarily accuracy. So a forecaster can be numerically wrong, but well aligned, or right for the wrong reasons, like I mentioned earlier.

60
00:09:02.750 --> 00:09:13.490
Ali Kore: And naturally, as everything else we've done in this bootcamp, the press releases are served in a cut-off-aware approach, so it sort of preserves the backtest validity as well.

61
00:09:15.260 --> 00:09:21.100
Ali Kore: And now, if we ran that, if we did run that judge over all 12 meetings, you would get this

62
00:09:21.270 --> 00:09:40.880
Ali Kore: this confusion square kind of a kind of a confusion matrix here where you get kind of a 2 by 2 matrix where you look at the whether the agent was right and the forecast was right, and whether the reasoning aligns. So if we look at this, we can see that 6 times the agent was right for the right reasons. So it made kind of the correct call, and the reasoning was aligned with the bank.

63
00:09:41.140 --> 00:09:49.309
Ali Kore: But if we kind of look at the off diagonal, you'll see that twice it was right for the wrong reasons, so it had the correct label, but the reasoning missed what the bank actually cared about.

64
00:09:49.560 --> 00:09:53.599
Ali Kore: And another two times, it was wrong, but its reasoning actually aligned with the bank.

65
00:09:53.980 --> 00:10:02.840
Ali Kore: And the interesting part is, if you only looked at, sort of, whether the forecast was right, like we did with the RPS metrics, you'd mislabel 4 of these 12 cases.

66
00:10:03.150 --> 00:10:22.739
Ali Kore: And we'll dive deeper into two of these cases. So on the left, you have June 2025, where the agent predicted a cut and the bank held, which was a wrong call. But the judge gave the reasoning a 0.85 because the agent correctly picked up things like the weakening labor market and inflation being near target, which is exactly what the bank also emphasized.

67
00:10:23.390 --> 00:10:34.319
Ali Kore: It just missed kind of the US tariff uncertainty as well. So that's what sort of dropped the reasoning score as well. So you'd have good reasoning, wrong outcome. And on the right, you look at the prediction for March 2026, where the agent correctly predict

68
00:10:34.450 --> 00:10:38.599
Ali Kore: but alignment was only of point 40, because it reasoned

69
00:10:38.760 --> 00:10:51.269
Ali Kore: steady state and inflation being a bit high, and missed a lot of the, geopolitical and energy drivers the bank actually cited for, their decision. So, there's another snapple of just, like, right answer, wrong reason as well.

70
00:10:51.740 --> 00:10:59.009
Ali Kore: And these are two cases where kind of the scoreboard that we originally had would grade these the same kind of as a lucky guess to some degree.

71
00:11:01.080 --> 00:11:12.720
Ali Kore: So, coming back to the primary point, this sort of points to an aspect of, I would say, like chain of thought, unfaithfulness, where a model stated reasoning often doesn't reflect what actually drove this answer.

72
00:11:13.170 --> 00:11:32.880
Ali Kore: So recent faithfulness work puts the rate at which models acknowledge the true cause for the reasoning fairly low. So a fluent, confident rationale isn't evidence necessarily of sound reasoning, and the only way to know is to evaluate the process against ground truth, which is exactly what the alignment judge does in sort of pulling in the bank's reasoning as well.

73
00:11:33.150 --> 00:11:42.789
Ali Kore: And it's kind of a good way to catch failure modes that underlie a lot of these aggregative scores that are more likely to hide these very intricate patterns of agent reasoning.

74
00:11:44.820 --> 00:11:52.160
Ali Kore: So, to summarize the overall, of what we've covered, one is kind of what we covered in terms of the score, so…

75
00:11:52.350 --> 00:12:03.109
Ali Kore: the baseline we have against climatology post cutoff and accept kind of the the result that we had in terms of the default agent and establish that sort of like that room to improve and where that lies in the next few presentations.

76
00:12:03.330 --> 00:12:10.590
Ali Kore: Judging the reasoning, which is the bulk of our presentation, by comparing the agent's rationale to some form of ground truth with an element as a judge.

77
00:12:10.700 --> 00:12:22.079
Ali Kore: And three, the approach where you sort of accept that correct… an agent being correct isn't always the same as an agent being aligned, and it's important to kind of evaluate the process you're going to deploy, not just the outcome.

78
00:12:22.280 --> 00:12:27.859
Ali Kore: And all of this kind of runs in Notebook 3, as well as Notebook 2, which goes into the background of, like, the

79
00:12:27.960 --> 00:12:31.200
Ali Kore: the benchmarking that we do against other conventional methods.

80
00:12:31.430 --> 00:12:44.130
Ali Kore: And this sets up kind of nicely… this sets up the new next two sessions nicely, where Ethan kind of takes an agent similar to this one, but actually optimizes it and uses exactly this kind of reasoning signal to improve it, in a feedback loop.

81
00:12:45.980 --> 00:12:55.109
Ali Kore: And that's all I have in terms of slides. So thanks for listening. I can take any questions now, and if there aren't any, I'll just move on to the the notebooks as well.

82
00:12:56.880 --> 00:12:59.380
Ali Kore: I think I see something in chat.

83
00:13:03.910 --> 00:13:06.150
Ali Kore: Okay, Ethan, answer that question. Cool.

84
00:13:07.220 --> 00:13:08.240
Ali Kore: And.

85
00:13:10.150 --> 00:13:11.650
Ali Kore: Here's another question.

86
00:13:11.870 --> 00:13:23.630
Ali Kore: How do we know that Lm. Is a judge is not wrong. Is it better to use the same Lm. For all of us just the one that provide the answer. Your reasoning different model. Well, that's a good question. I think you're right in saying that.

87
00:13:24.330 --> 00:13:25.440
Ali Kore: you,

88
00:13:25.950 --> 00:13:37.569
Ali Kore: the LLM is a judge… well, those are two different questions. I'll answer maybe the second one first, because it's an easier question, but I would say that you're absolutely right in saying that using a separate LLM as an LLM as a judge, particularly a strong

89
00:13:37.890 --> 00:13:45.849
Ali Kore: a much more capable model, but also something very different in terms of the class of model that you're leveraging as well. So using that

90
00:13:46.120 --> 00:13:46.949
Ali Kore: Okay.

91
00:13:47.080 --> 00:13:56.099
Ali Kore: using a different model to judge the model as doing the forecasting is a good is a best practice that you should follow always when you're sort of applying Lm. As a judge.

92
00:13:56.250 --> 00:14:15.719
Ali Kore: And this is for the purpose of just like ensuring that there isn't that kind of like alignment and reasoning that models tend to have nowadays where a model that you use the same model to sort of judge the reasoning of the old model. It's more likely if the model is going to accept that reasoning because it came from the model. It aligns a lot more with what the model would have said already.

93
00:14:15.970 --> 00:14:25.680
Ali Kore: So using a separate model is a good best practice to getting like a much more reliably, reliably reliable judgment on the models forecasting.

94
00:14:26.300 --> 00:14:32.890
Ali Kore: But the second question, the 1st question, I think, is a little more challenging and saying, how do we know the elements? The judge is not wrong.

95
00:14:33.440 --> 00:14:39.329
Ali Kore: I guess that would mostly come from kind of the rationale you would list it from the Lm. As a judge, and

96
00:14:40.400 --> 00:14:48.979
Ali Kore: I think there's an aspect in saying that the reason I stressed the idea of using like a very capable Lm. As a judge was on account of

97
00:14:49.210 --> 00:14:50.100
Ali Kore: sort of…

98
00:14:50.360 --> 00:15:02.989
Ali Kore: to some degree trusting, but also monitoring the rationale that the LLM is a judge. And when I say rationale, it's not the rationale of, like, the forecaster, that's a different rationale entirely, but we also elicit a rationale from the LLM as a jud

99
00:15:02.990 --> 00:15:10.910
Ali Kore: So you could investigate that rationale as well, and see if it aligns with the reasoning produced by both traces. So…

100
00:15:11.200 --> 00:15:12.910
Ali Kore: You could kind of…

101
00:15:13.130 --> 00:15:22.159
Ali Kore: reasonably trust. You can audit kind of on a human as a loop basis. What are the elements of judge producing, producing like the correct I would say, judgments.

102
00:15:22.520 --> 00:15:26.490
Ali Kore: Since you produced that rationale as well, so you could see if that's like a valid ration

103
00:15:26.700 --> 00:15:36.569
Ali Kore: But I'd say just having a very capable model that's different from the forecasting model would be a really good place to start in terms of defining an LLM as a judge that you can trust pretty well.

104
00:15:38.480 --> 00:15:41.289
Ali Kore: coming from the Cola. When Lm. Is a judge.

105
00:15:41.660 --> 00:15:57.509
Ali Kore: judges the predictions wrong to use the feedback to improve the prediction? Yeah, I'll leave that to Ethan, I guess. That's, like, the next step. It's, like, there's another two, like, a large number of, a large body of work within the bootcamp that covers that, so I'd say just, like, coming soon to that question would be cool.

106
00:15:58.070 --> 00:16:00.210
Ali Kore: Ground truth can mislead the church.

107
00:16:00.750 --> 00:16:03.730
Ali Kore: How did you approach this challenge?

108
00:16:04.610 --> 00:16:09.249
Ali Kore: Ground truth can mislead the judge. I'm not sure…

109
00:16:09.690 --> 00:16:13.339
Ali Kore: if I'm maybe misunderstanding that question in particular.

110
00:16:13.790 --> 00:16:18.200
Ali Kore: Could you elaborate on that, Lori? Around, like, ground truth misleading the judge.

111
00:16:20.640 --> 00:16:22.329
lurie Migalatii: Yeah, sure. Can you hear me?

112
00:16:22.330 --> 00:16:23.810
Ali Kore: Yeah, I can hear you. Thanks.

113
00:16:23.810 --> 00:16:37.629
lurie Migalatii: Yeah, I'm referring about ground truth or rubrics, how the judge would evaluate your answer. So it needs to have some ground truth that he can compare or run some metrics against.

114
00:16:38.670 --> 00:16:44.169
lurie Migalatii: So if ground truth is obsolete or, not updated fast enough.

115
00:16:44.510 --> 00:16:53.000
lurie Migalatii: Your judge would make can make a positive or negative assumption that the response makes sense and it's reasonable.

116
00:16:54.500 --> 00:16:58.070
lurie Migalatii: I guess someone… Somebody ask.

117
00:16:58.179 --> 00:17:01.769
lurie Migalatii: Kind of similar question, if you kind of provide the feedback.

118
00:17:01.969 --> 00:17:07.329
lurie Migalatii: In my case, it would be to your feedback piece to your ground truth.

119
00:17:08.510 --> 00:17:09.320
lurie Migalatii: Thanks.

120
00:17:09.709 --> 00:17:15.139
Ali Kore: Yeah, I'd say the ground truth in this particular instantiation, the way we set it up, is based on

121
00:17:15.729 --> 00:17:31.999
Ali Kore: reports published by the Bank of Canada, particularly their rationale. So it would be. It would be fair to say that the ground truth does become stale after a certain point in the context of back testing. So, having the most recent ground truth in a way that is sort of

122
00:17:32.229 --> 00:17:36.029
Ali Kore: specific to the time series that you're developing, I think, would be

123
00:17:36.399 --> 00:17:44.229
Ali Kore: I think would sort of offer a signal that is relatively stable. Like, I can't imagine that it would mislead the judge in terms of

124
00:17:44.389 --> 00:17:50.209
Ali Kore: having sort of like the most recent rationale for the bank decision in that case. But.

125
00:17:50.550 --> 00:18:06.019
lurie Migalatii: Yeah, my context was in the, what my question was is, like recent events, geopolitical, you say price jumps or tariff uncertainty last moment before you run by valuation. That's what I'm referring to.

126
00:18:06.550 --> 00:18:20.789
Ali Kore: Oh, you mean like changes in the sort of like environment post like or even pre when they're like pulling together a report for bank decisions themselves. Yeah, that's a good point I would say to sort of have a scenario where the model.

127
00:18:21.200 --> 00:18:31.800
Ali Kore: I guess that's something that Ethan can get into. I don't know what the, like, the guts of that system are, that self-improvement loop, but the ability to maybe reject that ground truth and say, actually, this isn't actually misalign

128
00:18:31.810 --> 00:18:46.319
Ali Kore: you know, this is a scenario where the bank itself was like slightly behind of like an evolving situation, and maybe the model in that case was more agile than the Bank of Canada. Yeah, that's a good question. I think maybe that's something Ethan will be better suited to answer.

129
00:18:46.790 --> 00:18:49.560
Ali Kore: and certainly the later presentations. Thanks, Larry.

130
00:18:50.210 --> 00:18:55.150
Ethan Jackson: I could jump in with a little bit of context here.

131
00:18:56.410 --> 00:19:12.699
Ethan Jackson: In no particular order, I think just, like, this question around LLM as a judge alignment and quality was something that we explored very deeply in the Agentic Evaluations Bootcamp, and, you know, we were able to spend an entire bootcamp kind of just asking the question, like, how well aligned is this LLM as a judge?

132
00:19:12.700 --> 00:19:16.669
Ethan Jackson: Then there's questions around feedback loops, which, yeah, we'll get into.

133
00:19:16.670 --> 00:19:22.380
Ethan Jackson: In the rest of this morning's session. And then as well, yeah, this is just, like, a little bit of a preview.

134
00:19:22.590 --> 00:19:30.020
Ethan Jackson: But I think in the context of forecasting or any kind of active decision process where an agent is is involved.

135
00:19:30.130 --> 00:19:35.450
Ethan Jackson: The way I'm interpreting your question, Larry, is, Say you get a resolution.

136
00:19:35.670 --> 00:19:42.629
Ethan Jackson: from a forecast, and it was right or wrong, or the reasoning was aligned or not.

137
00:19:42.890 --> 00:20:01.649
Ethan Jackson: If you have an agent that's equipped with a learning mechanism, how much weight should it put on that single like exemplar? Is it like something that should totally change its core methodology or is it just kind of like one, one data point that it should consider among many others and then just kind of like.

138
00:20:02.800 --> 00:20:13.279
Ethan Jackson: Two more, two more thoughts that came to mind, Ali, from your answers is like, right, there's this staleness question, like, just because, you know, if you calibrate an agent to reason like

139
00:20:13.420 --> 00:20:22.559
Ethan Jackson: the previous Bank of Canada analysts doesn't necessarily mean that that's going to hold over time. Like, their reasoning process may be changing. And so…

140
00:20:22.840 --> 00:20:25.520
Ethan Jackson: I think there's an opportunity to

141
00:20:26.600 --> 00:20:29.939
Ethan Jackson: Think of these agents as potentially having.

142
00:20:30.050 --> 00:20:40.620
Ethan Jackson: variable, configurable, adaptable strategies, and we could look for different ways to calibrate them based on information that's available.

143
00:20:40.900 --> 00:20:45.569
Ethan Jackson: Like, these analyst reports and all the context in the news and so forth.

144
00:20:48.080 --> 00:21:03.769
Ethan Jackson: and looking again at kind of like thinking, I guess, more broadly about what are the different ways that we want to evaluate them for alignment, because alignment for alignment's sake could also be misleading. I think maybe that was hiding in the details of these questions. Yeah.

145
00:21:05.890 --> 00:21:09.259
Ali Kore: Thanks, Ethan. Hongju, I think your question is like.

146
00:21:09.260 --> 00:21:14.889
Winnie Au: Ali, can I, can I just interject real quick? Because I, you still have to do the code walkthrough, right?

147
00:21:15.400 --> 00:21:16.180
Ali Kore: Yeah.

148
00:21:16.550 --> 00:21:27.449
Winnie Au: Okay, how about we just do that first, and then we address the questions just, like, by writing it, and then we can also talk, go through the questions, like, when we have, like, a bit of downtime. Hopefully that helps.

149
00:21:27.650 --> 00:21:29.899
Winnie Au: That way we don't run out of time. Yeah. Okay.

150
00:21:29.900 --> 00:21:31.699
Ali Kore: No worries, I'll share now.

151
00:21:32.960 --> 00:21:34.180
Ali Kore: Hmm, okay.

152
00:21:40.620 --> 00:21:49.109
Ali Kore: So… just run through some of these. Sorry. One sec. Zoom keeps popping up over the

153
00:21:50.250 --> 00:21:53.849
Ali Kore: Okay, so I'll just run through these notebooks pretty quickly.

154
00:21:54.090 --> 00:21:57.979
Ali Kore: The flow of, like, the overall notebooks tend to align,

155
00:21:58.400 --> 00:22:04.250
Ali Kore: with a lot of the other previous notebooks we've shown before, so they have a lot of the configuration set up and everything.

156
00:22:05.580 --> 00:22:20.499
Ali Kore: So I'll kind of skip over that quickly and show that this is kind of the scenario where we set up the benchmarking across the different models for this particular Bank of Canada rate direction setup. So this discrete event prediction.

157
00:22:20.970 --> 00:22:29.989
Ali Kore: So if we look at the 1st few cells, it's just mostly set up establishing kind of the back test configuration experimental config as well as the the back test spec.

158
00:22:30.210 --> 00:22:32.560
Ali Kore: Similar to other notebooks, like I mentioned earlier.

159
00:22:32.870 --> 00:22:34.600
Ali Kore: And this is kind of the…

160
00:22:34.700 --> 00:22:42.499
Ali Kore: What we consider kind of the warm-up, where you have the binary special case, where you kind of have a binary cut versus not case, and you sort of establish

161
00:22:42.520 --> 00:22:59.160
Ali Kore: that you can score that with Breer. And then we do a kind of a quick numerical check to check that that Rps with 2 categories tends to equal beer and kind of justify that notion that I mentioned in the slides where, like Crps Breer and Rps are just one big family of metrics.

162
00:23:00.990 --> 00:23:13.680
Ali Kore: And then we establish the predictors like we do in the previous slides. So you have the baseline here as well as the logistical regression, and then the Lmp. And the agent as well, the analyst agent.

163
00:23:14.720 --> 00:23:30.469
Ali Kore: and initializing those with the standard kind of like Gemini 3.1 flashlight model and establishing some of the other configs. And then we just run the entirety of the back test across the origins here. There's nothing particularly special here in terms of the new code or code you haven't seen before in previous notebooks.

164
00:23:30.970 --> 00:23:33.959
Ali Kore: But if I just scroll past, maybe…

165
00:23:34.660 --> 00:23:41.310
Ali Kore: here to look at, I would say, kind of these predictive distributions over time. You have

166
00:23:42.310 --> 00:23:53.920
Ali Kore: some of the plots, some plots that differ from what we saw in the papers, where we sort of like, look at what a mean score we sort of get. Look under the hood into what a mean score tends to hide in this case. So

167
00:23:54.090 --> 00:23:55.100
Ali Kore: And.

168
00:23:55.720 --> 00:23:58.210
Ali Kore: It's a scenario where I would say

169
00:23:59.630 --> 00:24:18.500
Ali Kore: we're looking at effectively the predicted distribution over cut, hold, and hike by the meeting. So you can see the kind of the bias each of these different models have. It's a useful plot for establishing how the model tends to, I would say, distribute the probability across these different

170
00:24:18.550 --> 00:24:22.559
Ali Kore: I would say, binary decisions.

171
00:24:25.240 --> 00:24:32.379
Ali Kore: and then a few of the other plots that we saw in the the notebook as well. Looking at the prediction versus the outcome.

172
00:24:34.030 --> 00:24:48.479
Ali Kore: where we look at kind of the the different models and how they predict the predicted probability in like a different way of establishing kind of the the rationale in terms of what the bank does and how the the models produce the rationale as well.

173
00:24:50.190 --> 00:25:03.239
Ali Kore: And a lot of these are just rehashed from the — a lot of this was pulled into the slides as well, but it's effectively the code that underlies the necessary head-to-head evaluation across all these models as well.

174
00:25:04.500 --> 00:25:11.829
Ali Kore: What I really wanted to get into was the this notebook, which is kind of the Lm. Is a judge alignment as well. So

175
00:25:12.220 --> 00:25:15.069
Ali Kore: This is kind of the process metric here where we look at the

176
00:25:15.190 --> 00:25:18.949
Ali Kore: The qualitative eval leveraging LLM as a judge, so…

177
00:25:19.170 --> 00:25:22.719
Ali Kore: There's a bit of setup, and we run the,

178
00:25:22.960 --> 00:25:31.630
Ali Kore: The Lm. Is a judge with the the line trace tracking as well. So these Lmp. And agent over the same 12 meetings.

179
00:25:32.030 --> 00:25:42.040
Ali Kore: And each of these models tend to produce their own rationale as well as a set of signals. So looking at them head to head creates a scenario where you can have a —

180
00:25:42.310 --> 00:25:54.159
Ali Kore: rationale sort of like alignment setup different from the other models. Since these models are Lms at the heart of it. They're they're both Lm processes at the heart. So you can just sort of at least with that rationale as well.

181
00:25:54.650 --> 00:26:03.390
Ali Kore: And then this, we sort of judge each trace. So for each of these traces we have kind of a strong judge that we

182
00:26:03.630 --> 00:26:09.209
Ali Kore: a sort of initialize that judges the bank's own press release for the meeting as ground truth like I mentioned earlier.

183
00:26:09.360 --> 00:26:19.600
Ali Kore: And it shows kind of that same slightly lower fidelity version of that same plot that I showed earlier where we have lean alignment as well as how many of them were correct as well as aligned.

184
00:26:21.510 --> 00:26:37.709
Ali Kore: And then we also have kind of the breakdown where we render it as markdown as well. So each verdict kind of links to his Langfuse trace. So you have a scenario where you can look at the signal overlap that the model produces, as well as the rationale for why it gave this particular alignment score.

185
00:26:39.170 --> 00:26:54.649
Ali Kore: And all these, tend to, be linked to Langfuse traces as well, so if I click this, it'll open in a different browser somewhere, a different screen, but you could essentially open the trace on Langfuse and have those, scores integrated directly

186
00:26:54.650 --> 00:26:59.399
Ali Kore: into the Langfuse trace as well, similar to how you would have any other metric.

187
00:26:59.410 --> 00:27:04.739
Ali Kore: integrated to Langfields, which is particularly useful when it comes to integrating it into a single place.

188
00:27:05.940 --> 00:27:09.939
Ali Kore: I think that's all I have in terms of the notebook. So

189
00:27:10.370 --> 00:27:13.820
Ali Kore: Thanks for listening. I guess. Move on to the.

190
00:27:13.820 --> 00:27:29.050
Winnie Au: Yeah, awesome. Ali, I think we can squeeze in one quick question, because there's some question in the Q&A. Someone is asking, just wanted to clarify for this implementation, how do you enforce the cutoff date that is abided by the LLM?

191
00:27:29.270 --> 00:27:30.709
Winnie Au: Thank you.

192
00:27:30.710 --> 00:27:33.350
Ali Kore: Sorry, say that again one more time.

193
00:27:33.350 --> 00:27:35.400
Winnie Au: Sure, I can also paste it in the chat.

194
00:27:35.500 --> 00:27:41.850
Winnie Au: Just wanted to clarify for this implementation, how do you enforce the cutoff date as abided by the LLM?

195
00:27:44.030 --> 00:27:48.319
Ali Kore: on… In this case, the cutoff dates

196
00:27:48.910 --> 00:28:05.560
Ali Kore: The cutoff dates are, since we're not doing any search in this particular case, the cutoff dates are provided mostly, heuristically from the upstream modules, so the data modules, that feed the data into the LLM. So as you're constructing the backtest, the cutoff date is aligned, it's sort of like a date time.

197
00:28:05.680 --> 00:28:19.400
Ali Kore: heuristically, rather than something that the LLM abides by. Since there's no search, and it can't go into web search, and that's something that Ethan gets into as well, you can't really… it's very, fairly simple to sort of, like, treat it as a data set where you can just

198
00:28:19.400 --> 00:28:28.110
Ali Kore: Apply that cutoff date heuristically to say you only consume the bank reports as well as the time series data up to that particular date.

199
00:28:29.920 --> 00:28:34.050
Winnie Au: Great, thank you, Allie. I think that's all the time we have, so we'll pass it over

200
00:28:34.200 --> 00:28:35.859
Winnie Au: Thank you so much.
