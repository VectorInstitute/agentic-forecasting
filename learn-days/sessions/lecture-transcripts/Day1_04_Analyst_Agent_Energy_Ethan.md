# Day 1 — Analyst Agent // Energy Markets Reference Implementation

**Outline session:** Presentation of "Analyst Agent" (reference code exec agent w/ agent skills) // energy markets reference implementation part 1 (11:30am–12:00pm) — Ethan Jackson. Includes trailing Q&A before lunch.
**Source file:** GMT20260708-152138_Recording.cutfile.20260708213426193.transcript.vtt

---

WEBVTT

1
00:00:00.120 --> 00:00:01.360
Ethan Jackson: to agents.

2
00:00:02.190 --> 00:00:05.180
Ethan Jackson: to agents that can fetch and compute.

3
00:00:06.000 --> 00:00:14.249
Ethan Jackson: So what we'll cover in this part, again, is kind of exactly that segue, how we transition from an LLMP as kind of like a static

4
00:00:14.290 --> 00:00:29.020
Ethan Jackson: context-fed prediction machine to an agent that is actively sourcing, fetching information, computing, and even tomorrow we'll talk about how they can learn.

5
00:00:29.260 --> 00:00:37.110
Ethan Jackson: But we still tie this to the same predictor interface so that we can compare these head to head against the other methods.

6
00:00:37.250 --> 00:00:43.940
Ethan Jackson: We'll talk about the anatomy and capabilities of, of this agent, the components,

7
00:00:44.270 --> 00:00:55.540
Ethan Jackson: the, kind of, like, the staircase of which features are, are, brought in, depending on, on what, on what you want to run. And we'll do a live, live demo of this agent.

8
00:00:55.580 --> 00:01:13.079
Ethan Jackson: The third thing is that we'll be able to look at a failure mode for this agent and kind of why, you know, LLMPs were kind of bad enough in terms of data leakage, but the problems become more subtle and even harder to pin down, I think.

9
00:01:13.080 --> 00:01:16.639
Ethan Jackson: When you start dealing with, with Agentic forecasters.

10
00:01:16.870 --> 00:01:23.780
Ethan Jackson: But then on the flip side, what we'll look at is just how versatile and flexible these Agentic predictors are.

11
00:01:24.340 --> 00:01:26.490
Ethan Jackson: So let's…

12
00:01:27.240 --> 00:01:44.130
Ethan Jackson: let's, let's keep going. The… the main difference is, like, and I think we all kind of, here in this call at least, have a good sense of what the difference between an LLM and an agent is, and it's very similar for an LLM process and an agent, but I think

13
00:01:44.250 --> 00:01:50.730
Ethan Jackson: I want us to think a little bit more creatively about what those differences could mean. So, as we saw, the LLM process

14
00:01:50.910 --> 00:01:57.540
Ethan Jackson: is a machine where the context is packed into the prompt for the LLM. You choose what it gets to see.

15
00:01:57.540 --> 00:02:17.230
Ethan Jackson: And you go from prompt to forecast. The agent, on the other hand, is responsible for taking initiative and executing actions to go and get the context. It can search, compute, decide. This is, you know, think of it as we're building the classical react loop around an agent, equipping it with tools and so forth.

16
00:02:17.710 --> 00:02:25.930
Ethan Jackson: But what we wanted to do for this bootcamp is kind of go straight to the modern interpretation of agents, and think

17
00:02:25.950 --> 00:02:42.190
Ethan Jackson: more along the lines of like, say, code executing agents, agents with skills. These are very, very similar to the ones that you could configure and could literally configure in, like, say, Claude Code or Codex, Cursor, and so forth.

18
00:02:43.160 --> 00:03:03.080
Ethan Jackson: But one thing I want to emphasize here is that what we saw in the LLM process presentation and what's in the reference implementations is a very bare bones interpretation of what it means to pack context into the prompt and reflecting on the question from an audience member earlier.

19
00:03:03.250 --> 00:03:15.579
Ethan Jackson: what would we expect to happen if you used Fable to power an LLM process? I would answer the question in a very kind of abstract way, and say that, like, well.

20
00:03:15.720 --> 00:03:35.169
Ethan Jackson: Fable probably isn't the model that you want to use to be the Llm. Process. But just think about how powerful Fable could be as an agent to decide which context to pack into the Llm. Process. And that's kind of our jobs as data scientists and engineers here, too, is you could. You could

21
00:03:35.520 --> 00:03:50.359
Ethan Jackson: be a lot more creative in thinking about what kind of context to pack into the LLM process prompt, it does start to blur the lines between what is an LLM process and what is an agent, but we'll keep coming back to this idea of, like.

22
00:03:50.420 --> 00:04:05.280
Ethan Jackson: what is the right context that you actually want going into the LLM, whether it's packed there in a kind of a pipelined way, or whether it's dynamically retrieved by an agent. We can think about these things as not totally separate.

23
00:04:08.570 --> 00:04:16.099
Ethan Jackson: So, moving on to agents, we'll define this kind of, simply as an LLM that can act, and…

24
00:04:16.269 --> 00:04:27.770
Ethan Jackson: for this purpose of this presentation today, this means searching for context, running code, and deciding when to emit a forecast. And this is still kind of concentrated around the predictor.

25
00:04:27.930 --> 00:04:35.400
Ethan Jackson: interface. Tomorrow, we'll expand this and spend a lot more time talking about other interpretations of agents.

26
00:04:37.440 --> 00:04:45.159
Ethan Jackson: Again, really just driving home the point that we are equipping our agents

27
00:04:45.330 --> 00:04:59.960
Ethan Jackson: with implementations of these predictor interfaces so that they can participate in the same experiments, like the ones that we just saw. The agent harness isn't really aware of this, or, like, the agent isn't aware that its harness is enforcing this.

28
00:05:00.110 --> 00:05:14.629
Ethan Jackson: And it's just to standardize everything so that we can have these consistent experiments. What you'll find, though, now is that as the code is getting a little bit more, complex as we're building agents.

29
00:05:14.920 --> 00:05:25.490
Ethan Jackson: There's a lot of organization around building agents that we have built into kind of methods that

30
00:05:25.960 --> 00:05:45.180
Ethan Jackson: build and configure an agent as a predictor. So you will see in quite a few places around the reference implementations that will kind of have like build predictor variants. And these kind of make it easy for us to bring in an agent configured as a predictor.

31
00:05:45.180 --> 00:05:49.030
Ethan Jackson: This means different things in different places in the code.

32
00:05:49.150 --> 00:06:03.840
Ethan Jackson: we try to give a little bit of separation there so that it kind of reads more naturally. Like, you know that you're building this agent for the purpose of participating in, kind of, an evaluation experiment compared to other predictors.

33
00:06:05.920 --> 00:06:21.950
Ethan Jackson: So now I want to take a look at the anatomy of this analyst agent that we have built. This agent appears in two reference implementations, the, the energy, reference implementation, as well as the Bank of Canada reference implementation.

34
00:06:21.970 --> 00:06:35.730
Ethan Jackson: And then there's a kind of an evolution of this, which is the adaptive agent that we will cover in tomorrow's sessions. This involves something called the strategy state, and we'll talk about this tomorrow.

35
00:06:35.910 --> 00:06:53.269
Ethan Jackson: Without that, this agent should have familiar components. We have an LLM core implemented using the Google ADK, Agent Development Kit, SDK, and this gives us, like, basically the React loop around the agent for free.

36
00:06:53.380 --> 00:07:11.579
Ethan Jackson: We equip it with, standard tools. One is for searching the web, and this is, calling a, it's like a tool that calls a sub-agent in a specific pattern, and that sub-agent is the Gemini grounded with Google Search.

37
00:07:11.580 --> 00:07:13.939
Ethan Jackson: So this is, like, Google's own…

38
00:07:14.060 --> 00:07:23.090
Ethan Jackson: kind of Agentic interface for doing web and news search under a combined interface, and it's… it works quite nicely.

39
00:07:23.720 --> 00:07:37.820
Ethan Jackson: The second kind of primary tool is for code execution. So we allow this code to write and execute its own code in the E2B sandbox.

40
00:07:37.820 --> 00:07:47.410
Ethan Jackson: The sandbox is best practice for Agentic code execution where you want to have security and potentially data privacy.

41
00:07:47.760 --> 00:07:51.199
Ethan Jackson: The way that we have built the sandbox is…

42
00:07:51.200 --> 00:08:08.839
Ethan Jackson: that it is automatically configured to import packages that are relevant for forecasting and kind of related to the environment. And it can do things like pull in data from StatCan. And if configured with API keys, it could pull in data from other places.

43
00:08:08.840 --> 00:08:11.530
Ethan Jackson: There's a variety of ways that this can be used.

44
00:08:12.380 --> 00:08:20.039
Ethan Jackson: And in fact, the kind of instructions And behaviors.

45
00:08:20.610 --> 00:08:25.609
Ethan Jackson: for which the agent should do code execution is defined via agent skills.

46
00:08:25.660 --> 00:08:43.009
Ethan Jackson: So we are kind of fully adopting agent skills as a paradigm in, both the analyst agent and for the adaptive agent. And if you're not familiar with agent skills, these are… I like to think of them as kind of, like, dynamic prompt sections.

47
00:08:43.010 --> 00:08:48.589
Ethan Jackson: That include instructions, data, code snippets, things like that, that…

48
00:08:49.060 --> 00:08:57.589
Ethan Jackson: You don't always want to be present in the system prompt, because then it might not be relevant for the exact thing that you're asking the agent to do.

49
00:08:57.590 --> 00:09:18.610
Ethan Jackson: And so we include skills here that kind of instruct the agent on how best to use the code sandbox if you're asking it to do statistical analysis or trend production or something else that's related to the forecasting task. But maybe you don't always want it to do that, so we leave it out of the system prompt, but it can fetch.

50
00:09:18.610 --> 00:09:20.899
Ethan Jackson: And load this dynamically as we see.

51
00:09:21.800 --> 00:09:36.240
Ethan Jackson: And then, really importantly, the other thing that we equip the agent with is an output schema, which, of course, varies depending on the forecasting task. But the short version is that we use Pydantic

52
00:09:36.340 --> 00:09:42.410
Ethan Jackson: type classes to define the expected response schema for a variety of activities.

53
00:09:42.520 --> 00:09:56.279
Ethan Jackson: in this bootcamp, and these agents are well-built to be able to adhere to those output schemas when they're required, such as for a prediction task. We wanted to output those well-formatted predictions.

54
00:09:57.180 --> 00:10:05.380
Ethan Jackson: There's another little trick here. I'll show you in a moment. But the other thing that you can do is like.

55
00:10:05.690 --> 00:10:16.220
Ethan Jackson: what you might notice is we have a remote tool for Search Web, and then we have remote code execution, and that's because this is machine-generated code.

56
00:10:16.250 --> 00:10:34.399
Ethan Jackson: Of course, you can define tools for your agent that don't have open-ended code generation, and one of the tools that we built, and I will demonstrate today, is a tool that lets the agent run an ARIMA forecast, and then get the result of that into context, which is a really cool pattern.

57
00:10:36.700 --> 00:10:42.949
Ethan Jackson: Okay, so next, we'll look at what does it mean to configure an agent.

58
00:10:44.210 --> 00:10:45.350
Ethan Jackson: You know.

59
00:10:46.170 --> 00:10:58.310
Ethan Jackson: The point here is that I'm presenting the analyst agent, but we can configure different instances of this based on the different capabilities that we equip it with.

60
00:10:58.330 --> 00:11:11.100
Ethan Jackson: At the very basic level, we could provide the agent with no tools, in which case it looks a lot like the LLMP, including the way the data are packed into the user prompt.

61
00:11:11.100 --> 00:11:22.999
Ethan Jackson: But then we can start to decide which tools we want to equip it with. We could give it any combination of, the news and web search, the code execution, plus the skills that, that…

62
00:11:23.030 --> 00:11:37.330
Ethan Jackson: help the agent understand how best to use it, and then we can have these kind of, like, locally executed tools. So our run forecast, lets the agent just kind of, like, you know, press an ARIMA button and then get the output.

63
00:11:37.500 --> 00:11:57.049
Ethan Jackson: But in this bootcamp, we could build any kind of tools like that. If you want to build a tool that runs specific pipeline of forecasts or data analyses, we absolutely can do that. And I think a huge open question in Agentic forecasting is exactly which tools would you provide to an agent?

64
00:11:57.050 --> 00:12:01.940
Ethan Jackson: Especially in terms of data analysis, that would improve its robustness.

65
00:12:01.940 --> 00:12:08.790
Ethan Jackson: So you're not just relying on the LLM to, to kind of, like, Spit out the numbers.

66
00:12:11.260 --> 00:12:19.370
Ethan Jackson: So I'll actually show a live demo here today, but the point here is that we can start to do head-to-head comparisons of

67
00:12:19.370 --> 00:12:42.360
Ethan Jackson: Agentic versus conventional methods. Here is a back tested comparison of auto ARIMA and news agent. Maybe the point here is to say that the agent's forecast is not necessarily more accurate in every case than ARIMA. But one thing that I do like is that overall the confidence intervals definitely seem to be more reasonable.

68
00:12:42.550 --> 00:12:44.060
Ethan Jackson: But…

69
00:12:44.320 --> 00:12:51.799
Ethan Jackson: I will switch over here to the code, just to give you a sense of what I'm talking about in terms of tool use.

70
00:12:52.630 --> 00:13:02.909
Ethan Jackson: So, over here, I'm in my cursor window, the experience should be quite similar using the Coder environment, with VS Code.

71
00:13:03.040 --> 00:13:06.950
Ethan Jackson: And what I'm going to show you is, under Analyst Agent.

72
00:13:07.290 --> 00:13:17.290
Ethan Jackson: what you'll notice is that we have the reference implementation notebooks, which I think of as the interfaces to the agent. This is how we're, like, you know, loading up backtests and

73
00:13:17.300 --> 00:13:28.920
Ethan Jackson: specifying which tasks we want to run in an interactive mode. But the underlying agent configuration is always in a kind of a sibling directory to these.

74
00:13:28.920 --> 00:13:42.339
Ethan Jackson: Here you can see I'm expanding the analyst agent directory. Over here is the adaptive agent that we'll see tomorrow. And I can go in and do different things here to configure how I want the agent to work.

75
00:13:42.620 --> 00:13:47.229
Ethan Jackson: And just to note that you don't… you probably don't need to change much

76
00:13:47.250 --> 00:14:04.240
Ethan Jackson: under these agent.py files, unless you want to modify system prompts and so forth. But if the level of configuration that you want to do is changing which tools are available to the agent, how data is passed into those.

77
00:14:04.240 --> 00:14:11.169
Ethan Jackson: You'll find kind of like knobs for all of those things within the reference implementation notebooks themselves.

78
00:14:12.350 --> 00:14:19.419
Ethan Jackson: Now, the next thing that I'll mention here is that if you go all the way to the bottom of any of these agent.py files.

79
00:14:19.980 --> 00:14:35.250
Ethan Jackson: There's a little method at the end which specifies which configuration of the agent you want to load by default if you're using the command line or the web interface for ADK.

80
00:14:35.400 --> 00:14:56.869
Ethan Jackson: So previously, I think the version that's pushed on main right now, if you do the like ADK run or ADK web, you'll just get the most basic configuration of the agent that doesn't really include anything in terms of tools. But I'm going to replace that for the purpose of this demo with a different configuration.

81
00:14:56.870 --> 00:15:00.840
Ethan Jackson: of this agent that includes, in addition, the search model.

82
00:15:00.840 --> 00:15:05.370
Ethan Jackson: And it includes the ARIMA forecasting tool.

83
00:15:05.770 --> 00:15:09.960
Ethan Jackson: So this agent can do… has a couple more tricks up its sleeve.

84
00:15:10.770 --> 00:15:15.700
Ethan Jackson: Then if I run… this in ADK Web.

85
00:15:16.290 --> 00:15:18.520
Ethan Jackson: We can load it up nicely.

86
00:15:18.730 --> 00:15:20.319
Ethan Jackson: in this interface.

87
00:15:20.880 --> 00:15:24.720
Ethan Jackson: We can select… the analyst agent.

88
00:15:24.880 --> 00:15:32.770
Ethan Jackson: The first thing that I'll do is just kind of ask for a little, description. Let's see, what tools and capa…

89
00:15:32.950 --> 00:15:35.120
Ethan Jackson: abilities do you have?

90
00:15:37.300 --> 00:15:55.660
Ethan Jackson: I show this for a couple of reasons. This interface, as far as I know, only works if you're running the code locally, but I did a quick check this morning, and it looks like we should be able to enable port forwarding to the coder environment. Don't hold me to it, but I will take it as a task

91
00:15:55.720 --> 00:16:11.060
Ethan Jackson: over the next couple of days to see if we can get this interface available for all of you during the bootcamp. It might just take me a minute. Otherwise, you can get an equivalent command line interface, it's just not as nice.

92
00:16:11.490 --> 00:16:28.429
Ethan Jackson: I'll show this for now. What's really cool is that we can, again, we can select which agent that we want to run. We can go into the trace here, and this will show us all the details about everything that the agent is doing as it handles a request.

93
00:16:28.620 --> 00:16:39.110
Ethan Jackson: So for a simple one, like just asking a description of the agent, it's just going to give us an answer. It'll tell us, okay, we've got a statistical forecasting tool, we've got the search web tool.

94
00:16:39.380 --> 00:16:56.840
Ethan Jackson: And its output schema is… is hard-coded for this agent to produce a probabilistic forecast, so it can… it can do this. Just for the sake of demonstration, what I want to show is… is something like this. Like, let's say, produce a 4…

95
00:16:56.950 --> 00:16:59.809
Ethan Jackson: Passed as of today.

96
00:17:00.180 --> 00:17:04.569
Ethan Jackson: July 8, 2026.

97
00:17:04.700 --> 00:17:10.889
Ethan Jackson: I… Let's say, at a horizon of… 2 weeks.

98
00:17:11.200 --> 00:17:19.510
Ethan Jackson: what… does an ARIMA forecast look like?

99
00:17:19.750 --> 00:17:21.469
Ethan Jackson: And do you?

100
00:17:21.849 --> 00:17:23.589
Ethan Jackson: agree with it.

101
00:17:25.780 --> 00:17:28.050
Ethan Jackson: So the point here

102
00:17:28.230 --> 00:17:37.889
Ethan Jackson: is to show a couple of things. One is that we can call this tool, the agent can call this tool dynamically to get a, conventional forecast.

103
00:17:38.050 --> 00:17:47.750
Ethan Jackson: The result is available here. It's a little hard to expand over these, but these little ARIMA forecasts are super, super fast to compute.

104
00:17:47.940 --> 00:17:59.070
Ethan Jackson: So it's going and doing that first, then it's searching the web for information related to, what's, you know, obviously going on in the Persian Gulf right now.

105
00:17:59.420 --> 00:18:03.010
Ethan Jackson: The situation is super volatile every single day.

106
00:18:03.180 --> 00:18:12.550
Ethan Jackson: And this is what it's looking for, for supporting information in terms of, in terms of an oil price forecast. That's what we might expect.

107
00:18:14.350 --> 00:18:20.370
Ethan Jackson: And then we get… a very detailed answer. So not only is it going to show us

108
00:18:20.500 --> 00:18:26.120
Ethan Jackson: A forecast… This is, you know, for two weeks ahead of today.

109
00:18:26.380 --> 00:18:32.389
Ethan Jackson: It will summarize what the ARIMA forecast is, give us the quantiles.

110
00:18:33.490 --> 00:18:38.199
Ethan Jackson: Then we have some information here that's completely live.

111
00:18:38.990 --> 00:18:42.690
Ethan Jackson: And then we have an opinionated

112
00:18:42.860 --> 00:18:46.859
Ethan Jackson: assessment. We strongly disagree with the ARIMA baseline forecast.

113
00:18:47.030 --> 00:18:52.969
Ethan Jackson: So it's calling out unmodeled supply risk.

114
00:18:53.310 --> 00:19:04.399
Ethan Jackson: saying things like, you know, immediate obsolescence, WTI is already trading at these elevated price levels, which is higher than, you know, this much of what the

115
00:19:04.550 --> 00:19:05.880
Ethan Jackson: prediction is saying.

116
00:19:06.700 --> 00:19:16.399
Ethan Jackson: And then it offers an adjusted view, so giving us its own point forecast with its own interpretation of what the confidence interval around that should be.

117
00:19:16.650 --> 00:19:21.340
Ethan Jackson: So this is exactly the combination of.

118
00:19:21.440 --> 00:19:27.490
Ethan Jackson: numerical analysis and contextual analysis that we want…

119
00:19:27.860 --> 00:19:33.179
Ethan Jackson: to see from an Agentic forecaster and want to evaluate.

120
00:19:33.180 --> 00:19:50.119
Ethan Jackson: And tomorrow what we'll talk more about is to calibrate this. Just because an agent can produce this analysis doesn't mean that it's necessarily better overall than if we were to just trust the ARIMA over time.

121
00:19:50.220 --> 00:20:03.119
Ethan Jackson: And, you know, I think we're starting to repeat ourselves a lot, but, you know, just think about the implications here. We want to backtest this, but how do you backtest the news? It's so difficult to do that.

122
00:20:03.200 --> 00:20:13.560
Ethan Jackson: But I'm going to talk about that a little bit, and we'll try to do our best, again, so that we can get at least an optimistic view of how well this system could work. But again, making no

123
00:20:13.680 --> 00:20:20.029
Ethan Jackson: Claim that there's any substitute for live evaluation of such a system that can go out and.

124
00:20:20.140 --> 00:20:22.310
Ethan Jackson: read information from the open Internet.

125
00:20:23.690 --> 00:20:28.300
Ethan Jackson: So let me go back to… the slides.

126
00:20:30.450 --> 00:20:37.160
Ethan Jackson: And I'll show you a little bit more about how we at least have tried to bridge the backtesting gap here.

127
00:20:38.640 --> 00:20:40.590
Ethan Jackson: So, we're talking about cut-offs again.

128
00:20:40.730 --> 00:20:53.620
Ethan Jackson: When we're talking about numerical methods and the time series data itself, we can enforce the cutoff. We have machinery in the repository to do that. It's… this is pretty easy to do.

129
00:20:54.820 --> 00:21:07.210
Ethan Jackson: on the web search side, we at least explored the question of like, well, what if we ask for a new search and send the cutoff date as a parameter like.

130
00:21:07.210 --> 00:21:19.640
Ethan Jackson: tell me what was going on in the world with respect to oil markets as of January 2026, for example. Does it leak information if we ask it to censor itself?

131
00:21:20.490 --> 00:21:33.660
Ethan Jackson: This is a very, very soft interpretation of a cutoff, but the point is that we can put a fence around the time series data, but we can't structurally put a fence around the open web. I know there are projects like GDELT,

132
00:21:33.660 --> 00:21:43.589
Ethan Jackson: And there are tools like Tivoli and other kind of, like, Agentic search tools. I tried passing them cutoff dates, they don't work. They just don't work.

133
00:21:45.150 --> 00:21:55.789
Ethan Jackson: So, what we did is we tried to implement, such a soft cutoff, and everything looked good for a while, but when I was running back tests, last week, I found all kinds of problems.

134
00:21:56.250 --> 00:22:00.860
Ethan Jackson: This is a smoking gun, and I'm, I'm very, like.

135
00:22:01.090 --> 00:22:09.819
Ethan Jackson: I don't know the word, but, like, unashamedly sharing this with you, because if you see something like this, I want you to recognize it immediately, too.

136
00:22:12.100 --> 00:22:20.399
Ethan Jackson: When I was running a previous backtest with the news agent, and this was just on 2026 data, so I'm not talking about the LLM being

137
00:22:20.720 --> 00:22:27.779
Ethan Jackson: polluted. I'm talking about the new search results coming back with the ground truths in them when they should have been censored.

138
00:22:28.050 --> 00:22:30.660
Ethan Jackson: Or redacted, if you want to call it that way.

139
00:22:30.830 --> 00:22:47.149
Ethan Jackson: Basically, what happened is that the results looked way too good to be true, and the prediction quality was dead on at every horizon. That totally broke the trend. It was out of distribution compared to what we'd seen before. It just, you know, classical machine learning kind of

140
00:22:47.150 --> 00:22:51.690
Ethan Jackson: overfitting evidence, but in this case, it was like daily leakage evidence.

141
00:22:51.790 --> 00:22:56.480
Ethan Jackson: I'll show you what we did to fix it, and by fix, I mean…

142
00:22:56.630 --> 00:23:14.970
Ethan Jackson: fix optimistically again. But at least once we built a better cutoff mechanism, the error went from this ridiculously low number to a more reasonable one. And then when we read through the traces, it looked okay.

143
00:23:16.460 --> 00:23:17.510
Ethan Jackson: But…

144
00:23:17.510 --> 00:23:39.669
Ethan Jackson: Yeah, so how we caught it again was kind of visual analysis of errors. It looked way too flat. It was way too out of distribution compared to the other methods. We went and read the traces to see what actually was coming back in the searches, and we could see that there was clear leakage of the eventual outcome.

145
00:23:39.760 --> 00:23:42.060
Ethan Jackson: in the search result.

146
00:23:42.510 --> 00:23:45.350
Ethan Jackson: So the traces were the key to finding that.

147
00:23:48.770 --> 00:24:00.929
Ethan Jackson: What we tried in order was asking the agent more firmly to please not do this thing where it fails to apply the cutoff. That didn't work.

148
00:24:01.170 --> 00:24:18.429
Ethan Jackson: The second thing that we did was to implement an independent verifier. I'll show you this in one of the next slides, but it's basically to put an evaluator in the loop during the new search, so that there's an independent check for leakage.

149
00:24:18.690 --> 00:24:25.020
Ethan Jackson: Is there any information in this report that could

150
00:24:25.440 --> 00:24:31.189
Ethan Jackson: plausibly be from, like, beyond the cutoff date. So that's…

151
00:24:31.420 --> 00:24:47.849
Ethan Jackson: better, it's not perfect. And then the third thing was a sneaky one, where there was, because the way the tool was built, we were relying on the agent, the outer agent, to pass the cutoff date as a tool parameter, and sometimes it just didn't do that.

152
00:24:48.000 --> 00:25:02.090
Ethan Jackson: if you don't pass the cutoff date, then it's like there's nothing, there's no information there. So we implemented that at the harness level, so that it's the cutoff date is always injected into the tool call, not leaving it up to the agent to decide to populate that.

153
00:25:04.200 --> 00:25:11.880
Ethan Jackson: How these pieces work together is as follows, and my, like…

154
00:25:11.930 --> 00:25:31.160
Ethan Jackson: I don't know if I'm just trying to save face a little bit here, or how to express that, but the flip side is, even if you don't think this is a reasonable way to enforce a cutoff, this is actually a really good example of a self-consistency solution. The kind of self-consistency that we're trying to achieve here is

155
00:25:31.380 --> 00:25:42.459
Ethan Jackson: arguably not possible to a rigorous degree. But if you're talking about putting evaluation or quality checks in the loop, this is a very good way to do it.

156
00:25:42.720 --> 00:25:49.140
Ethan Jackson: The way that it works is that we have the analyst agent, the kind of outer agent, calls the subagent as a tool.

157
00:25:49.530 --> 00:26:00.340
Ethan Jackson: This one uses the grounded with the Google search. And then we have an independent verifier, which is just a kind of an LLM wrapped with a prompt.

158
00:26:00.340 --> 00:26:09.660
Ethan Jackson: And it has a very specific task that it is supposed to do that I'll show you on the next slide. It basically is trying to enumerate

159
00:26:09.730 --> 00:26:23.830
Ethan Jackson: all the ways in which it could be possible that the search results contain contamination, and goes through a quick loop until the verifier is satisfied

160
00:26:24.390 --> 00:26:33.460
Ethan Jackson: past beyond the confidence threshold that the information is safe to return to the outer analyst agent. If it can't

161
00:26:33.970 --> 00:26:38.190
Ethan Jackson: Reach that level of confidence, it will fail loudly and stop.

162
00:26:39.100 --> 00:26:47.569
Ethan Jackson: If it succeeds, it will return the verified context, which can be supplied to to support the prediction.

163
00:26:52.880 --> 00:27:05.319
Ethan Jackson: Right. And so, if we look at, what was happening in the, prefix version, you could see that there was evidence of, like, actual prices, past the, cutoff date.

164
00:27:05.320 --> 00:27:16.379
Ethan Jackson: Again, I mentioned that this search, sometimes, skipped the cutoff date entirely, and, we also didn't have a specific trace link for this, so it was kind of hard for us to go and check.

165
00:27:16.380 --> 00:27:26.339
Ethan Jackson: after these fixes, we at least observed that these properties were met. It looked a lot more reasonable. But of course we can't be

166
00:27:26.340 --> 00:27:29.399
Ethan Jackson: entirely sure in a hands-off kind of way.

167
00:27:31.390 --> 00:27:32.240
Ethan Jackson: Okay.

168
00:27:32.370 --> 00:27:33.340
Ethan Jackson: So…

169
00:27:33.840 --> 00:27:49.560
Ethan Jackson: You can't always unleak a live backtest, and a live web index always knows how the story ends. So if we separate fence backtests from live runs, we might have a better chance of making things work.

170
00:27:53.230 --> 00:28:02.419
Ethan Jackson: I would say that this is kind of getting to the core tension of using Agentic methods for forecasting.

171
00:28:03.340 --> 00:28:11.750
Ethan Jackson: what we saw from ForecastBench is, like, the… the… if we trust the methodology of those and related… that benchmark and related ones.

172
00:28:11.890 --> 00:28:17.080
Ethan Jackson: LLM agents are the frontier. Like this, this, this is.

173
00:28:17.330 --> 00:28:22.879
Ethan Jackson: you, if we believe those methods and those those those leaderboards

174
00:28:23.050 --> 00:28:24.760
Ethan Jackson: That's what we would carry forward.

175
00:28:25.750 --> 00:28:29.200
Ethan Jackson: There's still some catching up to do with humans.

176
00:28:29.730 --> 00:28:37.040
Ethan Jackson: And especially for these highly contextual problems, but LLMs and LLM agents are clearly getting better at forecasting.

177
00:28:37.230 --> 00:28:45.729
Ethan Jackson: Backtesting LLM processes is hard enough. If you have a problem, that fits in that post-cutoff window.

178
00:28:45.850 --> 00:28:58.900
Ethan Jackson: and you're very sure that the LLM cutoff is, is, faithfully reported, perhaps an LLM process is something that you could, you could backtest, and relatively faithfully.

179
00:28:59.730 --> 00:29:09.510
Ethan Jackson: For agents, though, the whole hypothesis is that context is the key. And some of the most important context is coming from the open internet.

180
00:29:09.830 --> 00:29:18.669
Ethan Jackson: And so, if we want to rely on that kind of dynamic live context, I think it means that we really should be thinking about

181
00:29:18.920 --> 00:29:21.349
Ethan Jackson: a move towards live evaluation.

182
00:29:21.570 --> 00:29:26.839
Ethan Jackson: this is slow, because you're waiting for resolutions to come in.

183
00:29:26.950 --> 00:29:32.020
Ethan Jackson: Which makes your signal… Super sparse.

184
00:29:33.770 --> 00:29:45.819
Ethan Jackson: But the tension again there is that despite that signal sparsity, the evaluation science around this is still pointing towards these being the superior methods.

185
00:29:47.090 --> 00:29:52.520
Ethan Jackson: So, what I'm… what I'm encouraging people to do is… is if you are…

186
00:29:52.550 --> 00:30:10.079
Ethan Jackson: here to explore and learn about methods, and really want to understand, like, learn about building Agentic Forecasters with that goal in mind, maybe don't worry too much about live evaluation. If this is something that you're already doing in your organization, you know, you're already

187
00:30:10.170 --> 00:30:24.189
Ethan Jackson: running different kind of forecasting models, and you really want to know, like, okay, should I trust an Agentic Forecaster? I would encourage you to think about spending the bootcamp time setting up live evaluation, and that's something that we can help you with, for sure.

188
00:30:26.010 --> 00:30:31.370
Ethan Jackson: Just a couple more things about, about the agents themselves.

189
00:30:33.390 --> 00:30:42.039
Ethan Jackson: A lot of this has to do, again, I said before about configuration. And we can go well beyond what is included.

190
00:30:42.040 --> 00:30:59.980
Ethan Jackson: We have skills for statistical analysis, for trend projection that you can enable in the code execution version of the agent using the ETB sandbox. We'll look a little bit more at the agent skill side of things tomorrow, so we'll leave that, on the side for today. But the,

191
00:31:00.620 --> 00:31:13.879
Ethan Jackson: the space of possible configurations for a forecasting agent is very open-ended. There's so many different ways that we could do this, and they each might contribute to improvements.

192
00:31:15.830 --> 00:31:30.999
Ethan Jackson: The other way to look at things is that we can go beyond the forecast, kind of like we did with the agent today. We're asking for more of a subjective analysis, but we can look at different formulations of problems.

193
00:31:31.040 --> 00:31:39.079
Ethan Jackson: even if they're more kind of numerically oriented. Today, we looked at the trajectory problem, or like, can you ask for kind of

194
00:31:39.080 --> 00:31:51.180
Ethan Jackson: forecasting over time. We can ask for binary, predictions to say, like, do you think that there's going to be a move in the price of oil that's, like, greater than, 10% over the next week?

195
00:31:51.190 --> 00:32:02.199
Ethan Jackson: That can be, elicited differently from the agent and scored differently. And then finally, we can… we can ask for… for different things, like, you know, how would your forecast change if you assume this?

196
00:32:02.480 --> 00:32:08.230
Ethan Jackson: That, of course, is much more difficult to evaluate, but not impossible.

197
00:32:11.150 --> 00:32:14.870
Ethan Jackson: A quick note is that when we can…

198
00:32:15.480 --> 00:32:27.710
Ethan Jackson: when we configure an agent, we can configure it, like I've mentioned a couple times, with a response model. Say we are doing something like a what-if scenario analysis, we can define

199
00:32:27.900 --> 00:32:32.969
Ethan Jackson: an experiment more easily. If we if we

200
00:32:33.090 --> 00:32:52.480
Ethan Jackson: formalize the question, or at least formalize the answer schema using a pydantic model like we have here. So for scenario analysis, you could be asking for things like these that include key drivers that still unstructured text. But it could be a mix of kind of like, you know, numerical

201
00:32:53.000 --> 00:33:00.630
Ethan Jackson: factors like ranges and unstructured and unstructured elements that we can evaluate post hoc.

202
00:33:03.580 --> 00:33:19.580
Ethan Jackson: And so what to take forward from this? Again, this notion that for the agents, the capabilities are all configurable. There are these common interfaces that enable them to take on different

203
00:33:19.580 --> 00:33:23.600
Ethan Jackson: capabilities while still participating in the same experiments.

204
00:33:23.900 --> 00:33:34.669
Ethan Jackson: Anything that we're talking about, whether it's, like, news, code, skills, tools, the whole universe of possible configurations can still kind of,

205
00:33:34.970 --> 00:33:38.010
Ethan Jackson: kind of coalesce around one predictor interface.

206
00:33:38.150 --> 00:33:38.920
Ethan Jackson: Ahhh.

207
00:33:39.350 --> 00:33:58.810
Ethan Jackson: Watch out for scores that look too good to be true. Be skeptical. I think we've said enough times this morning why we need to be skeptical. And then the final thought again I was mentioning earlier is that if you're thinking about running a live experiment, then it's probably the right thing to do.

208
00:33:58.820 --> 00:34:01.339
Ethan Jackson: and we can. We can help you build towards that.

209
00:34:02.170 --> 00:34:06.669
Ethan Jackson: All right. I think that will conclude my part today. Winnie, I'll pass it back to you.

210
00:34:06.800 --> 00:34:13.469
Winnie Au: Awesome. Thank you so much, Ethan, for a great presentation and demo. We do have 3 questions. I'm not sure if we have time to

211
00:34:13.489 --> 00:34:29.170
Winnie Au: go over all of them. Like, we could always, address the questions after lunch as well. But there was a question, that came up kind of early on in your presentation around, are we also going to cover a bit of Google search grounding inner workings and limitations?

212
00:34:29.260 --> 00:34:33.859
Winnie Au: It is a powerful tool, but we have seen results that are quite inconsistent, and then

213
00:34:34.360 --> 00:34:35.909
Winnie Au: a follow up on that.

214
00:34:36.230 --> 00:34:43.460
Winnie Au: Part question is that is there a date metadata for filtering to strengthen the limit on the date cutoff?

215
00:34:44.820 --> 00:34:57.549
Ethan Jackson: On cutoffs, first, it's like, yeah, there's, the… the forecasting context objects carry that information, and it's kind of, like, up to us to decide how to use that. So, like, I would just say, right, in one case.

216
00:34:57.550 --> 00:35:14.800
Ethan Jackson: we left it to the agent to decide to apply it, and it didn't. So then we built it into the harness. We can explore different aspects of that, depending on how you're constructing agents. We don't have any specific plans to go into any more depth, say, during the learn days on the grounding with Google Search.

217
00:35:14.800 --> 00:35:34.450
Ethan Jackson: Honestly, we're using it out of convenience. It is the most convenient way to get kind of news and web search together, kind of straight from the source. But what I would say is that it's a great opportunity for us to explore the evaluation side. If something that you want to do during this boot camp is to run a long, relatively long experiment.

218
00:35:34.450 --> 00:35:37.279
Ethan Jackson: and build a subjective

219
00:35:37.340 --> 00:35:54.749
Ethan Jackson: like, LLM as a judge evaluator to look at the quality of what we're getting at an internal point during an agent trace. 100% we can do that. I ran out of time this morning to show you the LangFuse integration. Maybe I'll try to show that tomorrow, but we're very well set up to collect

220
00:35:54.750 --> 00:36:03.849
Ethan Jackson: traces and do analysis on them in the same style that we did during the AI Agents Evaluation Bootcamp previously.

221
00:36:04.810 --> 00:36:12.810
Winnie Au: Sounds good. There's another question in the, q and a for the LLM process forecast baseline for the analyst agent.

222
00:36:13.130 --> 00:36:20.929
Winnie Au: How much transparency do we have into how the forecast was arrived in terms of data, context, reasoning, et cetera, that was used?

223
00:36:21.430 --> 00:36:34.470
Winnie Au: So the same question applies to agents. We saw the results for a live WTI forecast in the demo. How much deeper than, the demo can we… we can go to analyze the decision-making process of the analyst agent?

224
00:36:35.890 --> 00:36:41.779
Ethan Jackson: I do think this is worth taking the last 2 minutes before lunch, Winnie, if that's okay?

225
00:36:41.790 --> 00:37:01.669
Ethan Jackson: The answer is Langfuse again. So all of our LLMPs and agents running through the notebook code are configured to upload their traces to your team's Langfuse. And what you can do is go into those traces. I'm not sure.

226
00:37:01.980 --> 00:37:21.970
Ethan Jackson: which one that is. But this is from one I ran earlier today, and you can see the full details of everything in one place. If I go in over here, you can see the system prompt and all of the contents. So it's like, here's how we're loading in the data. Here's how the quantiles are being passed in like

227
00:37:23.330 --> 00:37:33.990
Ethan Jackson: everything is visible here in one place. And then, of course, what we can help you do is kind of… the code is disjoint, it's not like we have

228
00:37:34.020 --> 00:37:53.920
Ethan Jackson: Top to bottom single scripts for implementing specific agent configurations. There's too much complexity for us to be able to do that. But all of the pieces are there. We'll have the system prompt, the code that packs the data, how the context is built and supplied. It's all traceable but

229
00:37:53.920 --> 00:37:57.220
Ethan Jackson: Easier to parse visually if you just look at the traces.

230
00:37:58.480 --> 00:37:58.823
Winnie Au: Sounds good. Thank you so much, Ethan. We do have two more questions in the chat, but we're thinking maybe you can just type the questions, and then after lunch, we can kind of go over them, in the period right before facilitation, because we do have, like, around 10 minutes of buffer time.
