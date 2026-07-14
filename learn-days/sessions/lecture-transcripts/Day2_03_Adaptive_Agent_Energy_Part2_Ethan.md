# Day 2 — Adaptive Agent & Energy Use Case Part 2

**Outline session:** Presentation of "Adaptive Agent" // energy markets reference implementation part 2 (10:00–10:30am) — Ethan Jackson. Includes trailing Q&A before break.
**Source file:** GMT20260709-133607_Recording.cutfile.20260709194145223.transcript.vtt (part 2 of 2 — this recording covers two back-to-back topics)

---

WEBVTT

1
00:28:35.860 --> 00:28:46.870
Ethan Jackson: Great. Thanks, Ali. That was super, super insightful. It's nice even for me to see it all packaged together like that in this session. So let's keep going.

2
00:28:47.150 --> 00:28:51.950
Ethan Jackson: There's a lot of material to get through in general and today.

3
00:28:51.950 --> 00:29:10.229
Ethan Jackson: But what I want to begin with is kind of a note that I don't think I wrote this down explicitly anywhere in the bootcamp material, but I'll say it now, which is like, as we've been going through this progression from LLMP, kind of, you know, this very basic usage of the LLM.

4
00:29:10.230 --> 00:29:15.190
Ethan Jackson: Through to agents, now today, through to agents that can learn.

5
00:29:15.190 --> 00:29:35.100
Ethan Jackson: I think there's a subtle paradigm shift away from treating Agentic Forecasters just as yet another numerical method. Like the LLMP is really framed all around, can the LLM just be an alternative to fitting a machine learning model for the purpose of generating numerical predictions?

6
00:29:35.100 --> 00:29:37.169
Ethan Jackson: When you think of it that way.

7
00:29:37.270 --> 00:29:54.410
Ethan Jackson: you think of backtesting and experimentation around model design, model selection in a very similar way, but I think as we now move towards these, like, you know, highly capable frontier agents, that can learn from experience and so forth.

8
00:29:54.610 --> 00:29:58.399
Ethan Jackson: There's… There's not… the…

9
00:29:58.500 --> 00:30:11.590
Ethan Jackson: the analogy to machine learning, I think, totally breaks down. We talked about this a little bit yesterday in the sense that, you know, we have this tension with the cutoff, like the training cutoff data in the LLM anyway.

10
00:30:11.610 --> 00:30:20.990
Ethan Jackson: And so the question that I'm starting to try to discuss in this boot camp is, how should we evaluate agents?

11
00:30:21.090 --> 00:30:28.950
Ethan Jackson: I don't think that we should evaluate agents like their numerical methods. We should evaluate them like they are analysts.

12
00:30:29.640 --> 00:30:33.540
Ethan Jackson: You would never ask a human to go and backtest.

13
00:30:33.880 --> 00:30:41.690
Ethan Jackson: economic forecasting over the COVID period, and then judge them based on how accurate their forecast was.

14
00:30:41.890 --> 00:30:47.380
Ethan Jackson: Every human knows what happened during that period. It's not possible to unlearn that.

15
00:30:47.700 --> 00:31:03.110
Ethan Jackson: I think the same tension is what we're now starting to name and articulate with agents. It's already a problem with the LLM. And then yesterday, we already saw that the tension gets even, let's say.

16
00:31:03.260 --> 00:31:10.879
Ethan Jackson: tighter with agents that can go out and read the news. It's very, very hard to evaluate this thing where you're just, like.

17
00:31:11.190 --> 00:31:16.360
Ethan Jackson: Pretend that you don't know what's going on in the world right now, what would your forecast be?

18
00:31:16.540 --> 00:31:19.430
Ethan Jackson: It's not realistic. You wouldn't do this with a human.

19
00:31:19.850 --> 00:31:23.139
Ethan Jackson: Maybe you shouldn't be doing this with an AI agent.

20
00:31:23.590 --> 00:31:35.069
Ethan Jackson: so just think… you know, I want you to keep that in the back of your mind as you're thinking about, like, you know, what is the right way to build experiments using, these… the different methods that we're introducing. And then…

21
00:31:35.070 --> 00:31:45.659
Ethan Jackson: keep it in the back of your mind also as we start today to talk about adaptive agents, agents that can learn meta-optimization methods. We're going to talk a little bit later this morning about

22
00:31:45.660 --> 00:31:56.449
Ethan Jackson: ways that agentic systems themselves can be automatically generated and improved. And this makes it even more difficult to kind of keep that numerical

23
00:31:56.630 --> 00:31:59.989
Ethan Jackson: machine learning evaluation analogy in mind.

24
00:32:00.910 --> 00:32:04.450
Ethan Jackson: Okay, well, let's get into this part first. The adaptive agent. What is it

25
00:32:05.350 --> 00:32:18.399
Ethan Jackson: We're going to frame this as a evolution of the analyst agent into an agent that can learn, an agent that can learn from experience specifically.

26
00:32:19.100 --> 00:32:27.609
Ethan Jackson: What we have done is built an adaptive agent that has a mutable strategy, which is just written as a file.

27
00:32:27.610 --> 00:32:47.070
Ethan Jackson: And it reads this strategy before making any forecast. It's also equipped with specific mechanisms for adapting that strategy. But like we were just talking about in the discussion in the last session, there's definitely going to be a tension, like when should an agent self update based on what kind of feedback?

28
00:32:47.210 --> 00:32:54.230
Ethan Jackson: This is entirely configurable and the subject of a lot of research, and we'll just be able to scratch the surface on that today.

29
00:32:55.180 --> 00:33:14.399
Ethan Jackson: What we've built to begin to explore that idea is this mechanism that has evidence layers that use a particular kind of schema and scaffolding. You can think of this as an agent harness that governs the way in which this agent self-updates.

30
00:33:14.760 --> 00:33:15.680
Ethan Jackson: But it's…

31
00:33:15.790 --> 00:33:32.970
Ethan Jackson: Again, one of many possible designs. And then we'll kind of look at what this agent was able to do in just a very, very simple experiment that you can replicate by following the notebooks in the energy oil price reference implementation.

32
00:33:34.250 --> 00:33:51.489
Ethan Jackson: So again, we have simple extension or an evolution of the analyst agent. Whereas the analyst agent stayed, had a fixed strategy that we configure once in code. Its instructions never change between runs. We just kind of apply it evenly over time.

33
00:33:52.060 --> 00:33:58.420
Ethan Jackson: And it's the, you know, same strategy, same capability, same toolset, everything's the same every time.

34
00:33:58.640 --> 00:34:15.439
Ethan Jackson: With the adaptive agent, the strategy that it follows, its perspective, maybe even some of its kind of core beliefs about the system that it's monitoring, these are held in a mutable skill file, another agent skill that we'll be able to take a look at.

35
00:34:16.410 --> 00:34:20.299
Ethan Jackson: It can update from its own forecasting experience.

36
00:34:20.830 --> 00:34:26.370
Ethan Jackson: We've only built this in a very simple way, and extending this to more sophisticated

37
00:34:26.690 --> 00:34:41.839
Ethan Jackson: Kind of machinery is definitely something that we can explore in the bootcamp. We'll share more details later. But it's… you can think of this as the same analyst agent, with a slightly different skill set, but now it has this, experiential memory attached to it.

38
00:34:44.250 --> 00:34:54.220
Ethan Jackson: reviewing the architecture diagram that we saw yesterday for the analyst agent, we're really just adding this strategy state. And…

39
00:34:54.820 --> 00:34:58.410
Ethan Jackson: What we have is an agent's skill.

40
00:34:59.050 --> 00:35:13.190
Ethan Jackson: that enables this core agent to be able to modify a strategy state. And we'll look at how this works in practice. But this is kind of the cool thing.

41
00:35:13.910 --> 00:35:16.460
Ethan Jackson: actually a very common pattern. If you

42
00:35:16.960 --> 00:35:41.309
Ethan Jackson: If you were to go into Claude Code or Claude Desktop or any of the other kind of desktop frontier AI interfaces, those agents are equipped with skills. And one of the default skills, at least in Claude, is basically a skill creator. So the out of the box agent is packed in with a skill to be able to create and modify.

43
00:35:41.310 --> 00:35:49.879
Ethan Jackson: its own skills. That's the pattern that we're following here. And we've just kind of like implemented a highly opinionated version of it that's geared towards forecasting.

44
00:35:50.980 --> 00:36:10.139
Ethan Jackson: This works in conjunction with conventional tools, ones that are implemented and run locally on the machine adjacent to the agent. But we also still have the code execution service, including the sandbox, so that AI-generated code is running in a secure sandbox environment.

45
00:36:10.850 --> 00:36:15.199
Ethan Jackson: Otherwise, it's pretty much the same thing that we saw yesterday.

46
00:36:17.680 --> 00:36:30.529
Ethan Jackson: Now, the tagline here is that now we have a machine that is accumulating experience and not just making predictions based on context alone.

47
00:36:30.700 --> 00:36:44.820
Ethan Jackson: What this can result in is a highly subjective version or instance of an agent where, depending on exactly the experience that it has accumulated, that will differentiate it from another instance of that agent.

48
00:36:45.260 --> 00:36:55.460
Ethan Jackson: It uses observations, hypotheses, and corrections from its own experience to carry this experience over in a consolidated way from session to session.

49
00:36:56.070 --> 00:37:05.830
Ethan Jackson: So the strategy effectively creates a surface, a learning surface for the agent that acts as its memory.

50
00:37:09.090 --> 00:37:20.180
Ethan Jackson: And to dive a little bit deeper, we have to look at how this memory should be structured. How is this going to be governed? What can the agent update? And what does each update demand in terms of evidence?

51
00:37:22.110 --> 00:37:33.840
Ethan Jackson: What we've built is a kind of something like a Bayesian update machine, where the idea is that updates would be based on observations.

52
00:37:34.140 --> 00:37:39.169
Ethan Jackson: hypotheses, calibrations, and changes to the approach narrative.

53
00:37:40.330 --> 00:37:53.839
Ethan Jackson: The agent has tools to record an observation so that it can register the idea that it has seen something over consecutive experiences, and then it can open a hypothesis.

54
00:37:54.520 --> 00:38:08.669
Ethan Jackson: So if the agent is starting to notice certain patterns in the data, it could open a hypothesis that it can keep in mind as it is accumulating consecutive experiences with the idea that

55
00:38:08.940 --> 00:38:10.340
Ethan Jackson: in order to…

56
00:38:10.450 --> 00:38:26.670
Ethan Jackson: solidify or graduate a hypothesis, there needs to be enough accumulated evidence. This is something that we can actually build into the agent harness, and it's preventing a situation where, like we were discussing before, like, a single outlier experience could

57
00:38:26.670 --> 00:38:44.039
Ethan Jackson: result in the agent kind of overwriting unreasonably what was otherwise a reasonable strategy. So, we avoid the situation where a single piece of feedback can carry a disproportionate amount of weight, and this is implemented actually in code that's wrapped around the agent.

58
00:38:44.180 --> 00:38:50.490
Ethan Jackson: Of course, what I have to say is that this is, like, this is one possible design. I will,

59
00:38:50.820 --> 00:38:53.759
Ethan Jackson: get to consider alternatives later on.

60
00:38:54.570 --> 00:38:57.779
Ethan Jackson: Taking a little bit more at this harness code.

61
00:38:58.000 --> 00:39:07.529
Ethan Jackson: So, again, wrapped around this agent, we actually have this, like a machine check to make sure that in order for,

62
00:39:07.530 --> 00:39:25.120
Ethan Jackson: a hypothesis to be graduated, and for it to be kind of codified in the agent strategy, it has to pass certain conditions. It's very, very simple here, but the scaffolding is, is totally extensible, so, like, you could experiment with this and say, like, well, if you had a much, much more conservative

63
00:39:25.460 --> 00:39:29.210
Ethan Jackson: Update mechanism. You could enforce that here.

64
00:39:31.920 --> 00:39:36.260
Ethan Jackson: So let's take a look at what this agent has actually learned.

65
00:39:36.360 --> 00:39:51.290
Ethan Jackson: So far. And again, this is based on a… just a one-run, experiment that you can, directly replicate by running notebooks, 5 and 6 under the, energy oil reference implementation.

66
00:39:52.930 --> 00:40:04.750
Ethan Jackson: I'll show a demo of this in a moment, but I want to lay out how the notebook sequence works. So the first step starting in notebook number 5 is to load context from 2025.

67
00:40:04.780 --> 00:40:17.339
Ethan Jackson: So we have 52 weekly news summaries that they can be pre-populated using a script, and the price history for, some data leading up and into 2025.

68
00:40:17.800 --> 00:40:29.170
Ethan Jackson: And we kind of frame this as the context that's available for an untrained adaptive agent that's very similar in design to the analyst agent that we saw yesterday.

69
00:40:29.310 --> 00:40:42.660
Ethan Jackson: Then we have a prompt that's written in the notebook code. You can change it, you can write anything you want, but this prompt is basically asking the agent to do some self-directed study. So it's kind of like me asking the agent to

70
00:40:42.660 --> 00:40:51.379
Ethan Jackson: Go and try a bunch of experiments related to volume changes in the period of 2025, and

71
00:40:51.750 --> 00:40:59.480
Ethan Jackson: Update your beliefs based on what your forecast… your forecasting experience looks like iteratively over that backtest period.

72
00:41:00.490 --> 00:41:20.409
Ethan Jackson: We don't tell it how many experiments to do, what kind of experiments to do. What it does is based almost entirely on, I would think, on the skills that you provide it with. So it's like if you're providing it skills to do an ARIMA forecast or an XGBoost forecast or something like that, that's going to bias.

73
00:41:20.410 --> 00:41:25.299
Ethan Jackson: the work that that agent will do. Agents will be biased towards using the tools that they have available, of course.

74
00:41:25.380 --> 00:41:29.869
Ethan Jackson: but it's, again, really open-ended by design.

75
00:41:30.550 --> 00:41:45.789
Ethan Jackson: throughout the process of that agent doing work, it will record its findings, log observations, open hypotheses, only using the tools and interfaces that we provide it. It may attempt to graduate some of those hypotheses if it has accumulated enough evidence.

76
00:41:45.790 --> 00:41:57.429
Ethan Jackson: And then to kind of close the loop on a simple experiment, we can run the notebook 06 in that implementation to compare on the out-of-sample

77
00:41:57.430 --> 00:42:09.060
Ethan Jackson: period of 2026. Basically, has this agent learned anything by studying the past that improved its ability to carry forward accurate predictions into the future.

78
00:42:09.380 --> 00:42:18.469
Ethan Jackson: So we'll take a pause here to go to the code so you can see this agent a little bit more in action. I'll start with the notebook code.

79
00:42:19.000 --> 00:42:27.510
Ethan Jackson: just to make this a little bit more concrete. So again, in this 05 adaptive agent training, this will reiterate what I was

80
00:42:27.640 --> 00:42:42.019
Ethan Jackson: just describing that we have a version or an instance of this agent that's configured with a kind of a blank slate. There's nothing really included in its strategy. And then the output from running this notebook

81
00:42:42.020 --> 00:42:48.629
Ethan Jackson: will be a trained version where the strategy document will look different.

82
00:42:48.920 --> 00:42:57.569
Ethan Jackson: So if I actually go into the untrained version, we can go into this skill directory for the adaptive agent.

83
00:42:57.680 --> 00:43:08.140
Ethan Jackson: and look at its… skill.md, and that defines the strategy. And so, it has some baseline, opinionated

84
00:43:08.700 --> 00:43:18.429
Ethan Jackson: strategy, based on different horizons. But then you'll notice here that it has, you know, there's no calibration, no open hypotheses, no observations.

85
00:43:20.000 --> 00:43:26.669
Ethan Jackson: And so if we Go back over here. I just want to show.

86
00:43:27.960 --> 00:43:32.210
Ethan Jackson: The, the prompt that we're giving to this agent.

87
00:43:32.780 --> 00:43:41.000
Ethan Jackson: So there's a system prompt behind the scenes, but what's more important is the actual task that we're giving it. This is the task

88
00:43:41.000 --> 00:43:53.080
Ethan Jackson: that defines more or less this open-ended self-exploration that I've assigned to this agent. You can write anything you want here, and the agent will then execute this task

89
00:43:53.080 --> 00:44:02.849
Ethan Jackson: And over the course of that task execution, the idea is that it will potentially, if it has enough experience, go and update the skill file.

90
00:44:03.580 --> 00:44:14.789
Ethan Jackson: So let me show you a little bit of a demo here. Again, if I go to this agent's runner, you'll see that there's code here to help you

91
00:44:15.040 --> 00:44:22.289
Ethan Jackson: get this off the ground and running. So 1st I'll run the version of the untrained agent.

92
00:44:25.730 --> 00:44:29.239
Ethan Jackson: I can open this up in the ADK viewer.

93
00:44:29.370 --> 00:44:36.240
Ethan Jackson: And I'll just say, briefly, Describe your strategy.

94
00:44:39.020 --> 00:44:44.730
Ethan Jackson: This shouldn't take too long, because it will just be reading the strategy file.

95
00:44:45.530 --> 00:44:53.020
Ethan Jackson: And… giving an output. So it basically say, my strategy is designed to be self-directing.

96
00:44:53.450 --> 00:44:57.000
Ethan Jackson: But there's nothing specific going on here because it hasn't learned anything.

97
00:44:57.850 --> 00:45:02.200
Ethan Jackson: After running, this notebook.

98
00:45:03.700 --> 00:45:07.739
Ethan Jackson: You'll have… this strategy trained file.

99
00:45:08.530 --> 00:45:14.220
Ethan Jackson: and I can show you kind of, you know. Pull the the finished cake out of the oven.

100
00:45:17.090 --> 00:45:18.729
Ethan Jackson: And show you this one.

101
00:45:19.190 --> 00:45:24.100
Ethan Jackson: And then I'll just ask the same… question.

102
00:45:24.380 --> 00:45:29.959
Ethan Jackson: Briefly, describe your strategy.

103
00:45:32.840 --> 00:45:39.360
Ethan Jackson: Should do the same thing and load the skill. Yep, load the trained version of the skill.

104
00:45:40.610 --> 00:45:59.230
Ethan Jackson: And what you'll see is that it has a current active hypothesis. Backtesting has shown that in high volatility environments, standard linear trend extrapolation significantly underperforms flat trend. I'm actively monitoring this pattern to calibrate whether flat trend models should automatically override trend predictions.

105
00:45:59.310 --> 00:46:18.979
Ethan Jackson: So this is really, really interesting to me, at least, this idea that it has just from a single task generated a hypothesis. Maybe this is something that we could dig into. And the interesting thing for me, too, is that you could go back to this notebook.

106
00:46:19.170 --> 00:46:32.569
Ethan Jackson: and run very clean experiments where you do one prompt at a time to try to give it the right learning task, and how to systematically evaluate that. An alternative thing that you could do

107
00:46:32.780 --> 00:46:47.579
Ethan Jackson: is to continue to interact with this agent and just assign it different tasks. I won't do that right here because it'll be very long running, but I could say, like, what if you try applying this to 2024 data? Does the pattern… does your hypothesis hold there as well?

108
00:46:47.730 --> 00:47:02.249
Ethan Jackson: Things like that. You could ask it to consider different kinds of forecasts. Like, what if you do an ARIMA? Can we learn to do a calibration on top of ARIMA forecasts that improves our performance during backtesting?

109
00:47:02.580 --> 00:47:03.620
Ethan Jackson: Stuff like that.

110
00:47:04.500 --> 00:47:18.750
Ethan Jackson: The point, though, would still be to do this iterative experimentation over the backtest and do very sparing evaluation over the recent kind of what's intended to be the withheld evaluation set.

111
00:47:18.750 --> 00:47:24.750
Ethan Jackson: So that you're always keeping something out of sample for evaluation. As you can imagine.

112
00:47:25.090 --> 00:47:31.689
Ethan Jackson: It could be very, very easy to overfit this agent to the backtest period.

113
00:47:32.060 --> 00:47:34.320
Ethan Jackson: So keep that in mind as well.

114
00:47:34.760 --> 00:47:39.819
Ethan Jackson: But this is, this is, I think, a very, very

115
00:47:40.630 --> 00:47:46.349
Ethan Jackson: interesting and powerful way to start to explore agents that can learn.

116
00:47:47.500 --> 00:47:48.300
Ethan Jackson: Okay.

117
00:47:49.190 --> 00:47:51.340
Ethan Jackson: So when I go back over here.

118
00:47:51.720 --> 00:47:53.029
Ethan Jackson: to finish this deck.

119
00:47:55.910 --> 00:48:08.980
Ethan Jackson: To summarize some of what this, this agent already found using that exact setup that I was already changing, this just visualizes this, basically the,

120
00:48:09.370 --> 00:48:18.940
Ethan Jackson: hypothesis, I should say, at this point, that, under elevated volume in terms of trading on, oil futures, that

121
00:48:18.990 --> 00:48:28.569
Ethan Jackson: the model, that was initially considered, didn't really hold, or, like, the method… the methodology didn't hold. In the elevated volume, the linear trend.

122
00:48:28.570 --> 00:48:39.709
Ethan Jackson: doesn't… doesn't work. It was actually quite a… quite a lot worse than just holding the last price. So this is, like, kind of a… an interesting, kind of refutation of the…

123
00:48:39.940 --> 00:48:47.730
Ethan Jackson: whatever the the default model perspective that this agent started with. So Let's learn something.

124
00:48:51.160 --> 00:48:56.439
Ethan Jackson: We already looked a little bit at this, of, like, how the,

125
00:48:56.720 --> 00:49:04.890
Ethan Jackson: strategy changed before and after running this kind of training curriculum, as in, I assigned this task to do some learning.

126
00:49:05.030 --> 00:49:09.159
Ethan Jackson: Basically after we have gone through that iteration.

127
00:49:09.160 --> 00:49:25.910
Ethan Jackson: The approach narrative didn't change, which is good, because we only gave it one task. It shouldn't fundamentally change the approach after one run. That's what this whole harness and, like, gating around the agent is supposed to do. But it did open one hypothesis.

128
00:49:26.400 --> 00:49:34.940
Ethan Jackson: that's exactly what we want to see. It's like accumulating some information. But it hasn't decided to fundamentally change its behavior just because of one

129
00:49:35.240 --> 00:49:37.020
Ethan Jackson: one run.

130
00:49:39.080 --> 00:49:49.320
Ethan Jackson: If you're interested, you know, the… on the out-of-sample withheld evaluation, there was a slight improvement after training, but, like, not at all.

131
00:49:49.330 --> 00:50:00.929
Ethan Jackson: close to statistically significant. So it's at least it didn't completely break the prediction quality. A slight improvement, but I would never call this

132
00:50:01.010 --> 00:50:02.850
Ethan Jackson: anywhere close to significant.

133
00:50:03.530 --> 00:50:05.570
Ethan Jackson: But it's stable, at least, which is good

134
00:50:10.680 --> 00:50:28.849
Ethan Jackson: Kind of along the same lines as what Ali was presenting this morning, what's very interesting to me is to be able to go and read the traces from these agents. So not just the skill file that it kind of self-edits over time.

135
00:50:28.950 --> 00:50:34.229
Ethan Jackson: But every agent run produces a trace, and we have the reasoning,

136
00:50:34.500 --> 00:50:40.940
Ethan Jackson: Tokens or what I should say more specifically is that we actually have a rationale

137
00:50:40.950 --> 00:50:59.680
Ethan Jackson: element which is produced during these forecasts, so that the agent is forced to always explain the rationale of its forecast. And these are things that we can both read for understanding, but we could also apply LLM as a judge, or other kinds of scalable

138
00:50:59.790 --> 00:51:05.829
Ethan Jackson: evaluation to to look for for different things that we might that we might be interested in finding.

139
00:51:06.300 --> 00:51:06.980
Ethan Jackson: So yeah.

140
00:51:07.440 --> 00:51:12.700
Ethan Jackson: But yeah, the cool thing is that it's like already learning and the rationale.

141
00:51:12.960 --> 00:51:18.960
Ethan Jackson: may be changing, even with just a small amount of evidence, even if the

142
00:51:19.560 --> 00:51:23.659
Ethan Jackson: overall prediction behavior is is changing more slowly.

143
00:51:26.160 --> 00:51:30.329
Ethan Jackson: So what we still haven't done, though, is

144
00:51:30.540 --> 00:51:36.239
Ethan Jackson: We haven't introduced what I will talk about in the next session as a validation gate.

145
00:51:36.500 --> 00:51:44.110
Ethan Jackson: If you notice that our… the harness that we built around this agent, Only requires…

146
00:51:44.550 --> 00:51:48.039
Ethan Jackson: A hypothesis to be kind of self.

147
00:51:48.550 --> 00:51:51.300
Ethan Jackson: Confirmed a handful of times.

148
00:51:51.470 --> 00:51:54.650
Ethan Jackson: before a strategy update takes place.

149
00:51:55.000 --> 00:52:00.370
Ethan Jackson: In other words, we're committing to updates without checking

150
00:52:00.650 --> 00:52:09.429
Ethan Jackson: for any kind of held-out improvement. This is a kind of feedback loop that we could absolutely build as part of the bootcamp, but we'll leave that

151
00:52:09.540 --> 00:52:13.390
Ethan Jackson: to to you. If this is what you're interested in.

152
00:52:14.530 --> 00:52:21.120
Ethan Jackson: Again, connecting to the next session, there's no archive of

153
00:52:21.120 --> 00:52:36.799
Ethan Jackson: past learnings being stored here. We have a single mutable skill file that is overwritten or edited in place over time, and that means that information may be lost over the course of running a long experiment.

154
00:52:36.940 --> 00:52:51.730
Ethan Jackson: Other methods that we'll see in a few minutes this morning specifically create an archive of past strategies so that information is never lost and can always be retrieved later throughout the course of the search.

155
00:52:52.220 --> 00:53:02.880
Ethan Jackson: And, another limitation of our, of what we've presented so far is that the schema, or the harness around the agent.

156
00:53:03.030 --> 00:53:19.499
Ethan Jackson: basically how the mechanism for learning is defined is very rigid. It's very opinionated and fixed. We defined this. There's no opportunity for the agent to kind of, like, change this part itself if it doesn't like the way that we've decided how it should learn.

157
00:53:19.780 --> 00:53:25.389
Ethan Jackson: and so, the… The summary here, though, is

158
00:53:26.800 --> 00:53:29.660
Ethan Jackson: We've built an agent that can change.

159
00:53:29.800 --> 00:53:31.100
Ethan Jackson: over time.

160
00:53:31.860 --> 00:53:33.690
Ethan Jackson: Does it actually improve?

161
00:53:34.830 --> 00:53:39.980
Ethan Jackson: That's the topic for the next section. I think we're right on time, so we'll take

162
00:53:40.310 --> 00:53:43.939
Ethan Jackson: I think we're going to take a break here.

163
00:53:44.620 --> 00:53:47.390
Ethan Jackson: Maybe I'll pass it back to Winnie for a time check.

164
00:53:47.760 --> 00:53:51.330
Winnie Au: Hi, hi Ethan. We have five minutes for some quick questions in the chat.

165
00:53:51.330 --> 00:53:51.940
Ethan Jackson: Great.

166
00:53:51.940 --> 00:53:52.819
Winnie Au: Let's do that.

167
00:53:52.960 --> 00:54:01.669
Winnie Au: So the first question is, this is a very fascinating topic. Is there any boot camp on just adaptive agent? I don't know. I can actually say that right now, but,

168
00:54:02.600 --> 00:54:22.160
Ethan Jackson: I will say, so we don't have this plan as a boot camp just yet, but the AI engineering team is, this is like, I would say, self-improving agents and systems for self-improving agents and all of the questions around that, like how to do it well, how to do it safely.

169
00:54:22.190 --> 00:54:32.720
Ethan Jackson: This is a major topic that my team is exploring on the AI engineering side, and just stay tuned. This is a big topic for 2026.

170
00:54:33.740 --> 00:54:41.849
Winnie Au: Awesome. Thanks Ethan. Next question is, is oversight applied to the instructions and memory updates dynamically? And if so.

171
00:54:42.060 --> 00:54:46.140
Winnie Au: how or relying on judging the outcome is sufficient.

172
00:54:46.650 --> 00:54:57.669
Ethan Jackson: So, no, we're not doing this right now. I'll say quite honestly that I didn't know how far to go with this reference implementation.

173
00:54:58.380 --> 00:55:12.440
Ethan Jackson: because it can… it can very quickly get overwhelming in terms of content, but also in terms of budget. We'll see that in the next… in the next section. So, what we… we tried to implement the most basic

174
00:55:13.220 --> 00:55:22.540
Ethan Jackson: kind of functional version or useful version of a self-improving agent, leaving a lot of these questions, like, to be explored

175
00:55:22.540 --> 00:55:33.949
Ethan Jackson: together. I would love if one of the bootcamp projects was basically to say, like, oh, we're going to try to do the adaptive agent, but we want to put a self-consistency module

176
00:55:33.950 --> 00:55:44.389
Ethan Jackson: On top of it, to really focus on the… the gate that you put around the learning mechanism to make sure, as well as you can, that that's robust.

177
00:55:44.860 --> 00:55:51.980
Ethan Jackson: For us, right now, there's no such guarantee, so… we'll learn by observing together.

178
00:55:53.200 --> 00:56:00.269
Winnie Au: Awesome. Thanks, Ethan. There's another question around what are the best practices for prompt creation in this case?

179
00:56:02.310 --> 00:56:12.249
Ethan Jackson: I don't know. What I… What I think is that.

180
00:56:12.510 --> 00:56:14.399
Ethan Jackson: If you were using…

181
00:56:14.610 --> 00:56:24.150
Ethan Jackson: a frontier LLM. Like, suppose that this agent that I was showing you was powered by Fable, or maybe we're gonna get GPT 5.6 today.

182
00:56:24.410 --> 00:56:25.380
Ethan Jackson: But…

183
00:56:26.460 --> 00:56:41.650
Ethan Jackson: Those agents or agents powered by those models tend to handle extremely high level kind of goal setting tasks much better than previous models. So I would be tempted to just say to those agents.

184
00:56:41.820 --> 00:56:44.879
Ethan Jackson: But… Go explore.

185
00:56:45.210 --> 00:56:57.579
Ethan Jackson: Just go learn what you can, come up with the best possible robust strategy for forecasting over backtesting, knowing that I'm going to evaluate you out of sample in the next step.

186
00:56:58.010 --> 00:56:58.749
Ethan Jackson: Yeah.

187
00:57:00.460 --> 00:57:02.490
Ethan Jackson: I think… for…

188
00:57:02.740 --> 00:57:08.780
Ethan Jackson: More mid-powered models, you want to be more specific about the task that you assign to the agent, and you might

189
00:57:08.970 --> 00:57:24.270
Ethan Jackson: want to just explore, like, you know, say, explore an idea around using ARIMA as a baseline, or volume trend analysis as a baseline, and explore different things like that. We could talk about this all day, though, so I'll hold myself to that.

190
00:57:24.270 --> 00:57:27.479
Isaque: No, thank you. Thank you, Ethan. I appreciate it.

191
00:57:28.590 --> 00:57:35.980
Winnie Au: Thank you, Isaac, for the question. The next question is, how important is harness versus model for adaptive agents?

192
00:57:36.320 --> 00:57:39.110
Ethan Jackson: This is the big question.

193
00:57:39.370 --> 00:57:50.090
Ethan Jackson: There is a theme in the field right now. I think that harness adaptation is kind of seeing a new

194
00:57:50.190 --> 00:58:09.050
Ethan Jackson: life again. A lot of what I was doing a couple of years ago, I would now call that harness optimization, building things like planner architectures where the reasoning models weren't really good at doing sequences of planning anymore. I think it's really important

195
00:58:09.090 --> 00:58:13.459
Ethan Jackson: For many reasons, but one specifically is that

196
00:58:13.580 --> 00:58:19.799
Ethan Jackson: Harness optimization and engineering can be very powerful for unlocking.

197
00:58:20.670 --> 00:58:31.120
Ethan Jackson: great capability, like levels of capability from less powerful, less expensive models, including open source models. Just a plug is

198
00:58:31.490 --> 00:58:47.510
Ethan Jackson: Some of the work that we have in our pipeline is to explore the question of, can we optimize the harness for an agent powered by an open source model, such that it can close the performance gap with a much more costly model?

199
00:58:47.520 --> 00:58:57.389
Ethan Jackson: And I think there's a lot of evidence from research that we're seeing, including from vector faculty research that indicates that there's a lot of untapped potential there.

200
00:58:59.290 --> 00:59:09.139
Winnie Au: Great. Thank you so much, Ethan. There are still a few more questions in the chat, so maybe you can just type in your answers and then we can take our break and be back here for 10:40 AM.

201
00:59:09.450 --> 00:59:11.440
Ethan Jackson: Yep. All right. Thanks, everyone.
