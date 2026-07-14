# Day 1 — LLM Processes // Canada's Food Price Report Reference Implementation

**Outline session:** Overview of LLM processes // Canada's Food Price Report reference implementation (11:00–11:30am) — Ali Kore
**Source file:** GMT20260708-150040_Recording.cutfile.20260708213246961.transcript.vtt

---

WEBVTT

1
00:00:08.232 --> 00:00:19.921
Ali Kore: So Banoush just showed the conventional methods that kind of set the bar for forecasting. And I'll be introducing a different kind of forecaster where we have kind of a large language model used directly as a probabilistic time series model.

2
00:00:20.162 --> 00:00:25.141
Ali Kore: For the most part, we'll kind of see what that means, kind of watch it beat some of the…

3
00:00:25.412 --> 00:00:36.302
Ali Kore: the conventional baselines on this Canadian food price task, and then we'll kind of spend some time on looking at what it means to kind of backtest an LLM on these kind of use cases.

4
00:00:38.702 --> 00:00:39.902
Ali Kore: So.

5
00:00:40.472 --> 00:00:52.972
Ali Kore: Going forward, we're going to cover three topics. We're going to cover what an LM process actually is, how it works with some of the real numbers we have from our fruit price report forecasting implementation.

6
00:00:53.192 --> 00:00:57.871
Ali Kore: And then kind of look at why these numbers are also an upper bound based on the the.

7
00:00:58.032 --> 00:01:04.702
Ali Kore: the sort of the training cutoffs that these LLMs have and kind of look at what honest evaluation might entail.

8
00:01:07.132 --> 00:01:08.232
Ali Kore: So.

9
00:01:08.502 --> 00:01:13.322
Ali Kore: When we think about can a language model forecast a number?

10
00:01:13.542 --> 00:01:28.272
Ali Kore: We're thinking about whether… not necessarily whether it can, like, talk necessarily about, like, inflation, but whether it can actually take a series of numbers and return a real predictive distribution with these calibrated uncertainties, which is what we get from the conventional methods.

11
00:01:28.422 --> 00:01:32.051
Ali Kore: And then we look at how we can score that head-to-head against ARIMA.

12
00:01:32.282 --> 00:01:38.091
Ali Kore: And the overall answer of this question is yes, and it leads us to…

13
00:01:38.252 --> 00:01:48.472
Ali Kore: consider what an LM process is. So, from the Rukima and Dubinod paper, last year, when you take a frozen off-the-shelf model without any post-training.

14
00:01:48.612 --> 00:01:51.781
Ali Kore: And you write the numeric history into the prompt as text.

15
00:01:52.112 --> 00:01:57.332
Ali Kore: That's basically covers, roughly what it means to use an LLM as a forecasting model.

16
00:01:57.592 --> 00:02:08.872
Ali Kore: And then you can kind of add what numerical models tend to not have, a lot of the language, built around this production, this, prediction task, so the units, the domain notes, you can add,

17
00:02:09.042 --> 00:02:24.212
Ali Kore: previous reports from the actual task itself, like the Canada Food Price Report itself, as a contextual… add context to the overall task. And then you can ask, and then you can have it elicit a full distribution, rather than just having it give a single number.

18
00:02:24.562 --> 00:02:32.341
Ali Kore: And a companion idea to this overall approach is the Context as Key Paper, where, like I mentioned, you can add free text context.

19
00:02:32.502 --> 00:02:47.712
Ali Kore: As an extra signal to the forecasting task. And the food price report that we'll get into in a later slide is exactly that kind of context where we… where we kind of, like, include this extra context to, leverage the… the real capability of this, LLM process.

20
00:02:49.712 --> 00:02:58.251
Ali Kore: So getting into the nature of the prompt itself, there's not too much complexity here. This is kind of close to a bridge version of our actual system and user prompt.

21
00:02:58.472 --> 00:03:03.982
Ali Kore: So we tell the model it's a probabilistic forecaster, and to return kind of the standard quantile grid that you would get.

22
00:03:04.172 --> 00:03:07.132
Ali Kore: anywhere between Q. 5 and Q. 95 is adjacent.

23
00:03:07.542 --> 00:03:22.772
Ali Kore: And then, to give it the extra context, naturally in that forecasting task, we serialize the last 30 months of CPI as text, and then we ask it for the next 12 steps. And again, it just returns JSON, and we can parse… using the quantiles, we can just parse that into a predictive distribution per month.

24
00:03:22.772 --> 00:03:29.001
Ali Kore: So, the best thing to take from this is that the entirety of the series, as well as the output, is just text to a prompt.

25
00:03:31.162 --> 00:03:41.932
Ali Kore: And I mentioned, like, the aspect of producing, like, a grid of quantiles, but, we leveraged two different ways to get the distribution out of an LLM, and we implemented both, in the bootcamp.

26
00:03:42.252 --> 00:03:49.702
Ali Kore: The 1st one is a set of sample trajectories. So you ask the model for a whole path several times, like multiple samples.

27
00:03:49.862 --> 00:03:53.152
Ali Kore: And then you read the quantiles off those samples, so…

28
00:03:53.492 --> 00:04:02.692
Ali Kore: this would resemble something closer to true sampling, but it ends up being n calls per origin as you continuously sample the LLM, so it ends up being very token-heavy.

29
00:04:03.482 --> 00:04:13.452
Ali Kore: The other method that aligns more readily with, like, if you're leveraging kind of a large-scale LLM, is to get the quantile grid in one shot. So it's one structured call that returns the quantiles directly.

30
00:04:13.622 --> 00:04:18.681
Ali Kore: This is a lot cheaper, and especially when you start prepending, say, large reports for context.

31
00:04:19.832 --> 00:04:37.701
Ali Kore: Because you only pay for that context once when you're producing the entire quantile grid, and then you can just produce the time series that way. This is implemented through the same predictor interface that I'll get into in the notebook. So it amounts to the same contract, but just like a different way of elicitating the distribution.

32
00:04:39.622 --> 00:04:40.852
Ali Kore: Oh.

33
00:04:41.092 --> 00:04:42.042
Ali Kore: So.

34
00:04:42.432 --> 00:04:58.501
Ali Kore: On the next slide, we're kind of going to look at one of the plots from the notebook, which is where we start at July 2023, and we forecast every month of 2024 for the overall food CPI. So the black line is what actually happened in the future, food inflation kind of flattening out.

35
00:04:58.582 --> 00:05:11.442
Ali Kore: And the pink line is the LLM process, where you get, kind of… we can show kind of the median plus the 80% band on the uncertainty. And the blue dashed line is AutoREMA, and you can see, based on the conventional method, AutoREMA can sort of

36
00:05:11.712 --> 00:05:17.421
Ali Kore: Tends to just extrapolate the recent upward trend, which is basically the only thing that it can do with the information that it has.

37
00:05:17.682 --> 00:05:23.232
Ali Kore: But the Llm. Process kind of anticipates the flattening stays close to the the true forecast.

38
00:05:24.902 --> 00:05:31.681
Ali Kore: And before we dig deeper into the results, I just want to, like, do a quick reminder of something Ethan introduced this morning.

39
00:05:31.802 --> 00:05:41.202
Ali Kore: Crps as a score kind of scores the whole predictive contribute distribution. So you think of it as an ma between the forecasts, cumulative distribution and the ground truth.

40
00:05:41.442 --> 00:05:46.532
Ali Kore: So something to take away is that it rewards being accurate, but also appropriately uncertain.

41
00:05:46.882 --> 00:05:49.932
Ali Kore: Lower is better in this case.

42
00:05:50.312 --> 00:06:08.101
Ali Kore: And now back to the task at hand. When we look across all 9 food sub indices with 6 years across, say, 6 years of July origins. you'll see that it beats the classical baselines. So auto rema comes out to around 4.9. The L. 1 process and different configurations come down to around 3.7 to 4.3.

43
00:06:08.302 --> 00:06:17.912
Ali Kore: So it seems that a frozen sort of general purpose language model with no training on the particular data beats the classical methods on genuine, genuine, sort of food price prediction task.

44
00:06:18.642 --> 00:06:37.402
Ali Kore: But now we kind of come to the sort of like the honesty problem that LLMs have in that a backtesting an LLM isn't necessarily very honest. So AutoREMA at a July 2020 at the origin can only see data up to that date. So it's cut off safe by construction in the way you train it. You can backtest it on any date forward.

45
00:06:37.422 --> 00:06:40.992
Ali Kore: But an LLM can't necessarily make that promise. So Gemini's training runs on

46
00:06:41.032 --> 00:06:44.091
Ali Kore: Training runs roughly to early 2025.

47
00:06:44.182 --> 00:06:49.351
Ali Kore: So when we back-tested on 2024, it may simply have read what 2024 food inflation turned out to be.

48
00:06:49.852 --> 00:07:02.162
Ali Kore: And it gets a little more subtler than that, where some models can report two different cutoff dates, a stated one, as well as an effective one, where knowledge actually kind of trails off, so you can't necessarily trust the cutoff, stated there either.

49
00:07:03.061 --> 00:07:09.012
Ali Kore: So the best way to consider a historical LLM score is as an upper bound, especially when you're doing backtesting in this way.

50
00:07:09.372 --> 00:07:25.312
Ali Kore: Not necessarily a benchmark on the performance. So the only honest score you can have, and I think it was something we alluded to earlier on in the other presentations, is data from well after the training cutoff. So you could define a protected window that the model couldn't have seen. And this is the only thing that can keep our comparisons, particularly meaningful.

51
00:07:27.012 --> 00:07:39.222
Ali Kore: And now we come back to sort of grounding it back altogether, sort of pulling it all back together into the food price support implementation. But first I want to sort of introduce what the Canadian food price actually is.

52
00:07:39.362 --> 00:07:45.791
Ali Kore: It's a fairly well-known annual publication. Dalhousie University has produced it with partner universities every year since 2010.

53
00:07:46.182 --> 00:07:51.942
Ali Kore: Each edition forecasts how much food prices will rise next year by these subcategories.

54
00:07:52.292 --> 00:08:05.211
Ali Kore: It's particularly a good example because it tends to be a blend of expert human judgment and kind of this evolving toolkit over time, over the years of, like, statistics, machine learning, and then, to sort of, like, the modern day, increasingly, LLM usage.

55
00:08:05.522 --> 00:08:14.872
Ali Kore: So it offers a good, clean, kind of scorable task to pit our methods against, as well as offering a considerable amount of context when you're performing the backtesting.

56
00:08:18.952 --> 00:08:26.582
Ali Kore: And then as for the actual forecast task itself, the food price reports itself publishes one number per category.

57
00:08:26.682 --> 00:08:33.121
Ali Kore: So next year's expected average, which is basically just next year's expected average over average percent change.

58
00:08:33.412 --> 00:08:38.961
Ali Kore: And across the nine categories, we produce that for the LLMs themselves.

59
00:08:39.132 --> 00:08:44.841
Ali Kore: So you get 12 monthly forecasts from a July origin collapsed into kind of like that headline year over year.

60
00:08:45.142 --> 00:08:57.011
Ali Kore: And here's kind of the LN process versus what actually happened in 2024. You'll notice that it's kind of, like, good on certain things, like meat. Pretty close on restaurants overall, but it overshoots several categories, particularly fruit and bakery.

61
00:08:57.312 --> 00:09:08.281
Ali Kore: So even though it wins on Crps, it isn't necessarily clairvoyant. And naturally, it's also good to remember that due to like the cutoff that I mentioned earlier. All of these numbers are still an upper bound on performance.

62
00:09:11.162 --> 00:09:27.342
Ali Kore: And coming to that aspect of context is key, especially where it becomes code. The Food Price Report itself is just a PDF, so you can extract it to text with a publication date and then drop it in a cutoff-aware document store and turn it on with a particular flag.

63
00:09:27.342 --> 00:09:32.382
Ali Kore: in the codebase. So, it gets sourced into the report, so at each origin, the store

64
00:09:32.472 --> 00:09:36.501
Ali Kore: Only surfaces reports published on or before that date for the purpose of backtesting.

65
00:09:36.602 --> 00:09:41.241
Ali Kore: And it works exactly like the series data. So you never actually leak a future report.

66
00:09:41.732 --> 00:09:48.802
Ali Kore: The only caveat is that for a historical year, the report tends to contain a section that basically contains the answer.

67
00:09:48.992 --> 00:09:53.882
Ali Kore: So you can measure the lift on post-cutoff origins, just not pre-cutoff ones.

68
00:09:57.512 --> 00:10:14.492
Ali Kore: And to wrap things up, just like three things to carry forward from this presentation. The first one being that an LM process tends to be — can work as a real probabilistic forecaster. And what we've shown on the food price report is that it can beat classical baselines.

69
00:10:14.752 --> 00:10:23.832
Ali Kore: And the second point is that we have to be disciplined, considering that the pre-cutoff score for an LLM is upper bound, and the only honest number you can get out of these models is post-cutoff.

70
00:10:24.392 --> 00:10:38.681
Ali Kore: And three, that the kind of open frontier is context, where you, wire reports into the prompts that are cut-off aware, and you kind of see where measuring, where they help is kind of something left as an open exercise. And everything I showed here kind of runs in Notebook 2.

71
00:10:38.862 --> 00:10:39.862
Ali Kore: I'm

72
00:10:40.212 --> 00:10:50.041
Ali Kore: and I'm kind of going to run through the notebook as well to show how it all evolves in code as well. But I guess I can take questions for now, but I'll share my screen and open up the notebook while we wait.

73
00:10:51.042 --> 00:11:02.711
Winnie Au: Awesome. Thank you. Ali, while you put up your notebook, there is a question in the q and a and a person is asking, is there a chance the newer models like Fable would do better on that?

74
00:11:02.872 --> 00:11:07.731
Winnie Au: This was just, you know, I think your previous slide.

75
00:11:08.842 --> 00:11:14.292
Ali Kore: Oh, questions whether people would do better. That's a good question.

76
00:11:15.062 --> 00:11:25.852
Ali Kore: I guess it depends on… I guess that gets really into, like, the Agentic side of things, because, the way we're kind of producing… introducing LLM processes, in particular, is that,

77
00:11:25.962 --> 00:11:34.712
Ali Kore: it sort of creates a really good segue going from conventional and then on to some of the later, work we have around, like, extending that Agentic capability.

78
00:11:34.992 --> 00:11:49.502
Ali Kore: So an LLM process is, like, almost like a precursor. So think of it as, like, a nascent, almost, like, somewhat agentic system, where you have, like, an LLM, I would say reasoning over some of the context provided by these food record supports, as well as,

79
00:11:49.542 --> 00:11:55.452
Ali Kore: being able to predict these raw numbers on these kind of quantiles and uncertainty estimation.

80
00:11:55.552 --> 00:12:07.652
Ali Kore: I would say once you get into, like, these, these state-of-the-art models, like Fable and whatnot, you have to consider the same aspect of cut-offs, because obviously Fable's, like, a newer model, and you would expect, considering,

81
00:12:07.792 --> 00:12:26.951
Ali Kore: the nature of the model that it would do better, but it would make more sense to leverage it in a more like, I would say, refined Agentic harness. So we'll see kind of in the future presentations that you would technically expect Fable to do better, but you would expect it to do better in a harness that's much more elaborate than just like this basic LLM process on its own.

82
00:12:28.422 --> 00:12:42.961
Ali Kore: these were more, like, designed for, I would say, like, smaller base models, where you elicit, kind of, these quantiles based on just, like, the next token prediction capability of the model itself, rather than, like, this large, capable model, kind of, reasoning over the future using multiple tools.

83
00:12:44.592 --> 00:12:48.702
Ali Kore: Oh, sorry, I'm supposed to share my screen here. Let me try that again.

84
00:12:48.702 --> 00:12:56.011
Winnie Au: Yeah, no worries. Awesome. Thank you so much, Ali, for answering the question. I'm just gonna monitor the chat for additional questions while you get set up.

85
00:12:57.522 --> 00:12:59.922
Ali Kore: One sec, I can't seem to find…

86
00:13:01.672 --> 00:13:06.191
Ali Kore: Always lose the share button. Let me. Okay, here it is.

87
00:13:06.442 --> 00:13:07.632
Ali Kore: Okay.

88
00:13:08.042 --> 00:13:09.162
Ali Kore: Perfect. Yeah.

89
00:13:09.772 --> 00:13:11.322
Ali Kore: Okay, let me.

90
00:13:11.442 --> 00:13:12.502
Ali Kore: whoops.

91
00:13:12.782 --> 00:13:25.861
Ali Kore: Okay, so this is the notebook that I mentioned earlier that covers kind of the experiments around the Canada Food CPI. And it mostly focuses on the LMP in extension to the conventional methods.

92
00:13:26.592 --> 00:13:38.461
Ali Kore: So here we have the standard set of the setup code that pulls together a lot of the harness, the standard inputs for the Agentic Forecasting Bootcamp.

93
00:13:39.752 --> 00:13:56.481
Ali Kore: And here we have the configuration code, which sets up — I think Benush kind of introduced it in her slides as well. But it sets up the config that drives the rest of the notebook. So we're on a particular configuration called mini signal, which is mini — sorry, mini recent.

94
00:13:56.562 --> 00:14:09.551
Ali Kore: Which is just 9 food categories around 6 July origins from 2019 to 2024. And that tends to span the COVID and inflation regimes as well. And then everything downstream of the notebook just kind of adapts to this particular configuration as well.

95
00:14:10.552 --> 00:14:23.682
Ali Kore: And here we conduct kind of that data exploration from the slides as well, where we look at the the breakdown of Cpi across all the sub indices, and you kind of see those other ones, the other categories I talked about, like bakery and meat.

96
00:14:24.372 --> 00:14:25.452
Ali Kore: And.

97
00:14:26.342 --> 00:14:37.042
Ali Kore: And here, we kind of define the task itself via this backtest specification. So this sort of defines, of course, sort of the origins as well as the horizons that you're predicting over.

98
00:14:37.252 --> 00:14:48.781
Ali Kore: It makes it so the eval is reproducible across both the conventional methods as well as the LMP, so they all, integrate into the same backtest spec, as defined by this particular, specification.

99
00:14:50.072 --> 00:15:06.722
Ali Kore: And here is kind of a key cell where we integrate a lot of the predictors that we're going to test in this notebook. And we look at the kind of the competitors here where we naturally pull in naive last value, auto REMA, as well as two LLM processes. We do both sample trajectory as well as quantile grid.

100
00:15:06.782 --> 00:15:13.771
Ali Kore: For sample trajectory, we'll use a slightly lightweight model, because it's relatively token heavy. And then for

101
00:15:14.312 --> 00:15:29.892
Ali Kore: the quantile grid will use something stronger, since we can afford more of the tokens there, since we're getting the quantile grid in one shot. So a lot of these configurations are already made in a way to sort of, like, provide a foundation for how you should, like, leverage these different recitation modes.

102
00:15:30.042 --> 00:15:31.832
Ali Kore: Effectively. Thank you.

103
00:15:33.422 --> 00:15:52.772
Ali Kore: As we scroll down here, this is just where we sort of have these helper scripts for sort of building the different LMP implementations. So you can see here just a Gemini 3.5 flash with a certain amount of history, a set of report that it sources, and this sort of under the hood will sort of establish the backtesting discipline as well in that regard as best it can.

104
00:15:54.392 --> 00:16:06.051
Ali Kore: And here's kind of commented out Agentic representation as well, but we sort of leave it commented out because we want to focus mostly on the difference between the conventional methods compared to the

105
00:16:06.302 --> 00:16:07.592
Ali Kore: Lmp.

106
00:16:08.932 --> 00:16:19.981
Ali Kore: And here is where we basically run the entirety of the backtests, so we run every prediction across every origin. It gets cached on disk, so you can just load it up instantly if you've already run the notebook before.

107
00:16:20.342 --> 00:16:28.832
Ali Kore: And here you can kind of watch the mean CRPS print for each predictor. And you can already see kind of like the LNPs are considerably lower.

108
00:16:29.572 --> 00:16:44.971
Ali Kore: And here's one of the slides that you saw from the presentation as well, where it sort of forecasts all of 2024, where you have black as the ground truth and then pink as the LM process. And then I think blue dash as well was the auto remote.

109
00:16:45.122 --> 00:16:50.082
Ali Kore: all of them kind of extrapolating the trend over that horizon.

110
00:16:52.312 --> 00:16:58.712
Ali Kore: And then just more of these plots that reflected over the the breakdown of different food price calculations.

111
00:16:58.862 --> 00:17:07.991
Ali Kore: And you can also look at, kind of, the average year over year, where you collapse those monthly paths into, kind of, that single headline number that the report, actually shows.

112
00:17:08.482 --> 00:17:11.581
Ali Kore: So next year's average over average.

113
00:17:11.742 --> 00:17:13.872
Ali Kore: percent change per category.

114
00:17:15.542 --> 00:17:26.162
Ali Kore: And this is the leaderboard from that other slide as well, where you look at the comparison for the CRPS, where a lower CRPS is better, and you can see LMP down around 3.7 to 4.3.

115
00:17:27.842 --> 00:17:38.261
Ali Kore: And here we have a separate plot where we kind of look at the MAPE per category.

116
00:17:38.692 --> 00:17:50.441
Ali Kore: it's a… it's a similar… it's a somewhat… it depends on, like, what metric you prefer, but it's a familiar metric that sort of offers a different lens where you have this percent error per category, rather than what CRPS represents.

117
00:17:51.232 --> 00:17:53.822
Ali Kore: And then we have kind of like another

118
00:17:53.942 --> 00:18:02.011
Ali Kore: set up a kind of a headline table for looking at the best predictor per category as well. So you look at the median year over year for all of them.

119
00:18:03.302 --> 00:18:11.861
Ali Kore: And then, finally, we have kind of a something that segues into that kind of like the next sort of like next thesis of the next talk. So

120
00:18:11.972 --> 00:18:21.081
Ali Kore: every… it sort of, comes down to the point where every score here is kind of pre-cut off, so naturally it's an upper bound. And the only clean test, like I mentioned, is to have live forecasting.

121
00:18:21.262 --> 00:18:32.141
Ali Kore: where you forecast a year that the model couldn't have read. And that's kind of the project that's going to extend out into the next few use cases as well as the the next few, I would say, like presentations as well.

122
00:18:33.082 --> 00:18:37.531
Ali Kore: And that's where we reach the end of the notebook. So I'll take any questions.

123
00:18:37.722 --> 00:18:39.072
Ali Kore: on as well.

124
00:18:40.742 --> 00:18:42.272
Ali Kore: CLAB, See if there is.

125
00:18:42.762 --> 00:18:45.292
Ali Kore: Anything in charts. Nope.

126
00:18:46.322 --> 00:18:47.601
Ali Kore: All right. Thank you.

127
00:18:49.582 --> 00:18:54.752
Winnie Au: All right. Thank you so much. Ali, Jesse, should we stop recording and then restart for you?
