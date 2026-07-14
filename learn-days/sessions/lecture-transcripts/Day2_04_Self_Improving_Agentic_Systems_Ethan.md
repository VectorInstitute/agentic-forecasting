# Day 2 — Self-Improving Agentic Systems

**Outline session:** Self-improving agentic systems micro-lecture (ADAS, Darwin Gödel Machine) (10:35–10:55am) — Ethan Jackson. Opens mid-sentence, continuing a Q&A from just before the break. Ends as recording is stopped to hand off to Jessee Ho ahead of the industry spotlight talk.
**Source file:** GMT20260709-144216_Recording.transcript.vtt

---

WEBVTT

1
00:00:00.000 --> 00:00:01.470
Ethan Jackson: pitch, or…

2
00:00:01.890 --> 00:00:15.420
Ethan Jackson: Just continuing along the discussion of the previous one, like, yeah, is there going to be programming on self-improving Agentic systems? I kind of took an opportunity to fold this into the Forecasting Bootcamp because I wanted, to share

3
00:00:15.830 --> 00:00:34.590
Ethan Jackson: All of these kind of exciting advancements in the field that maybe we haven't had an opportunity to kind of talk about them all in one place. And this bootcamp is a great place to do it. So let's just continue. I want to let you know, where does this adaptive agent that we're talking about fit into the research landscape, at least as I see it?

4
00:00:34.820 --> 00:00:59.080
Ethan Jackson: And there's a really cool arc of papers that you might not know. It's all related to one vector faculty researcher, Jeff Klune. His lab and his grad students have produced some really, really cool papers. Jeff Klune is one of the heavy hitters of evolutionary algorithms. I've been following his work since I was a grad student in 2014.

5
00:00:59.310 --> 00:01:04.069
Ethan Jackson: Back then it was a lot of evolutionary reinforcement learning.

6
00:01:04.069 --> 00:01:20.319
Ethan Jackson: The theme that he has been consistently publishing in for the last several years is the application of evolution-inspired algorithms to create systems that can self-improve agentic systems.

7
00:01:20.780 --> 00:01:34.339
Ethan Jackson: So, we have 3 papers in this arc. The first one was, ADAS, or Automated Design of Agentic Systems, and they introduced this idea that the design of an agent, like, of the prompt.

8
00:01:34.340 --> 00:01:44.250
Ethan Jackson: and the code that we put as the harness around the model, we can use LLMs to search for new iterations of those pieces, because

9
00:01:44.250 --> 00:01:52.670
Ethan Jackson: the LLM obviously can generate text, it can generate code, it can do both of those things. That means that it can generate an agent harness, and…

10
00:01:52.740 --> 00:01:55.910
Ethan Jackson: This has been quite successful.

11
00:01:57.230 --> 00:02:07.289
Ethan Jackson: The progression of that work to the DGM, or Darwin-Godel machine, introduces a few things, but the key point here is to

12
00:02:07.460 --> 00:02:13.740
Ethan Jackson: guide the search process so that only changes that improve

13
00:02:14.210 --> 00:02:18.589
Ethan Jackson: Your system in a specifically measurable way.

14
00:02:18.840 --> 00:02:22.049
Ethan Jackson: will be retained in the search process.

15
00:02:22.950 --> 00:02:31.180
Ethan Jackson: And going forward to a recent paper, Alma, this is Automated Learning of Memory for Agents.

16
00:02:31.450 --> 00:02:40.619
Ethan Jackson: Is to specifically apply this to learning the memory mechanism for an agent that is going to be operating in a temporal environment.

17
00:02:41.200 --> 00:02:59.680
Ethan Jackson: exactly what we're talking about here. And all of these converge on kind of one key point, which is something that we're missing right now that I mentioned earlier, which is a held-out validation gate. How do we want to build a gate so that we are only making changes to our system, that measurably

18
00:02:59.730 --> 00:03:05.830
Ethan Jackson: improve or are likely to improve our our objective quality.

19
00:03:07.950 --> 00:03:18.860
Ethan Jackson: Going a little bit more into the details of this paper. Atas introduces a meta agent, which is, I think this is a very common pattern now, that

20
00:03:18.860 --> 00:03:28.299
Ethan Jackson: writes the code and prompt of other agents. And that means that these can be powered by different models. You can have a frontier model that is writing the agent prompt and harness.

21
00:03:28.450 --> 00:03:33.419
Ethan Jackson: For an agent that is going to be powered by a less powerful model, and you could optimize

22
00:03:33.580 --> 00:03:37.750
Ethan Jackson: for, like, say, something like Pareto optimal cost performance.

23
00:03:37.940 --> 00:03:47.340
Ethan Jackson: And… The research shows that this consistently leads to better agent designs than handcrafted or human engineered ones.

24
00:03:47.690 --> 00:04:10.960
Ethan Jackson: The Darwin-Godel machine is taking this self-rewriting aspect to a further degree. I won't explain everything that's going on in that paper, but again, there's more emphasis placed on this gating mechanism for demonstrated performance improvements, and they keep a more structured archive of diverse solutions so that we're not just overwriting one

25
00:04:10.960 --> 00:04:13.639
Ethan Jackson: improvement at a time. We retain

26
00:04:13.970 --> 00:04:22.470
Ethan Jackson: potential improvements or potential innovations. Sometimes they're called into an archive, and the search is able to access those.

27
00:04:22.670 --> 00:04:30.029
Ethan Jackson: So we have kind of a version of the gate and the graduate hypothesis, but it's very, very basic, and we don't have an archive.

28
00:04:30.210 --> 00:04:34.359
Ethan Jackson: As you can imagine, an archive could grow

29
00:04:34.460 --> 00:04:50.130
Ethan Jackson: to be very large, very quickly, and that makes experiments using these kinds of methods where you might be considering things like, oh, what if I take this idea from one iteration, this idea from another iteration, combine them, generate a new,

30
00:04:50.130 --> 00:04:54.919
Ethan Jackson: potential solution, it can be very, very expensive.

31
00:04:56.750 --> 00:05:08.800
Ethan Jackson: Now, Alma specifically, this is really interesting. So again, thinking about the memory mechanism itself as a specific target for self-adaptation.

32
00:05:09.070 --> 00:05:19.599
Ethan Jackson: So if we think of statelessness in Llms like you, you know, we the way we understand if you just separate the Llm. From from an agent.

33
00:05:19.990 --> 00:05:20.700
Ethan Jackson: And then…

34
00:05:20.810 --> 00:05:28.520
Ethan Jackson: there's no memory, there's no state, there's nothing retained between prompts, even. You have to build all of that around the model.

35
00:05:28.700 --> 00:05:39.049
Ethan Jackson: So this is specifically trying to… trying to kind of… bridge that gap to having Purpose,

36
00:05:39.630 --> 00:05:42.030
Ethan Jackson: designed stateful.

37
00:05:42.870 --> 00:05:49.879
Ethan Jackson: AI systems that are Tailored for the specific use case that you're deploying it in.

38
00:05:51.140 --> 00:05:53.620
Ethan Jackson: And that's what that's what this is all about.

39
00:05:56.200 --> 00:05:59.910
Ethan Jackson: How is this related, and how can we kind of tie this together with forecasting?

40
00:06:01.700 --> 00:06:15.830
Ethan Jackson: Where what we've hand-designed is a specific schema for how memories should be formed, or how kind of experience should be logged. This observation, hypothesis, calibration, and overall strategy narrative.

41
00:06:15.830 --> 00:06:22.350
Ethan Jackson: This would not be hand-designed by Alma or any of these methods. It would be the subject of a search.

42
00:06:22.400 --> 00:06:31.379
Ethan Jackson: an Llm. Guided search where lots of different possibilities to the schema would be considered, and we would retain the ones that are

43
00:06:31.470 --> 00:06:33.939
Ethan Jackson: functionally fit in the environment.

44
00:06:35.060 --> 00:06:49.129
Ethan Jackson: In terms of retrieval, we also, again, have a hard-coded mechanism. The skill is loaded into context every time. That's just how this… how the agent skill works, so there's no…

45
00:06:49.750 --> 00:07:01.650
Ethan Jackson: There's no opportunity to modify that unless you rebuild the system. With what Alma could do is that this could be a dynamic search process where specific

46
00:07:02.040 --> 00:07:21.480
Ethan Jackson: parts of the strategy, for example, could be retrieved depending on what's going on in the context. And in fact, that entire retrieval process, again, is also subject to a search and optimization kind of machine. So that whole mechanism is meta-learned.

47
00:07:21.720 --> 00:07:25.170
Ethan Jackson: And again, the research shows that this is very powerful.

48
00:07:25.530 --> 00:07:41.660
Ethan Jackson: Same thing with the updated rules. For the update rules, we have a specific harness around the agent that governs this. Alma or these other methods would learn those update rules and use out-of-sample validation

49
00:07:41.790 --> 00:07:44.600
Ethan Jackson: to determine which ones to retain.

50
00:07:46.630 --> 00:07:50.090
Ethan Jackson: So again, just taking a look to make it concrete.

51
00:07:50.830 --> 00:07:56.549
Ethan Jackson: A big part of our schema is defined just as a Pydantic type class.

52
00:07:57.910 --> 00:08:10.070
Ethan Jackson: what tends to work well when you're doing Agentic search is to standardize interfaces as much as possible, and to lean on these kind of type class definitions, just so that you're kind of setting a common language for the search.

53
00:08:10.710 --> 00:08:26.029
Ethan Jackson: But this is exactly the kind of thing that would be automatically generated and, you know, alternatives would be considered dynamically. You might end up with something very different if you were to apply Alma. So this would be part of the search space, not something that you would write by hand.

54
00:08:28.500 --> 00:08:33.110
Ethan Jackson: I won't go over all of this just for lack of time.

55
00:08:33.270 --> 00:08:51.060
Ethan Jackson: but I wanted to present, this, architecture slide. It's very, kind of, common across the three papers that I was showing, where we basically have a meta agent, again, which could be frontier powered, which is

56
00:08:51.270 --> 00:09:01.409
Ethan Jackson: sampling the kind of state of progress so far. Like, what are the designs that we've considered? Like, what are the memory schemas or the update mechanisms that have worked well so far?

57
00:09:01.440 --> 00:09:19.370
Ethan Jackson: It will… it will propose a modification plan. It will go through a discrete planning stage. Then there's an implementation phase to implement that plan. So that could be the same LLM, or a different delegated LLM, and you end up with a new candidate memory design.

58
00:09:19.410 --> 00:09:38.590
Ethan Jackson: And then in the forecasting context, you would then very likely go and evaluate that over one backtest run, potentially a whole series of backtest runs. It really depends on how deeply you want to go. And this is where all of the expense comes in.

59
00:09:38.590 --> 00:09:43.259
Ethan Jackson: How many new memory designs do you want to consider, where for each one.

60
00:09:43.540 --> 00:09:59.940
Ethan Jackson: your evaluation requires a full backtest. One backtest of the oil price reference implementation, depending on what model you use, can cost… it can cost upwards of, like, $100, and that's for, like, the mid-Gemini 3.5 flash.

61
00:10:00.310 --> 00:10:06.749
Ethan Jackson: So I present this kind of saying, like, this is what the state of the art is doing.

62
00:10:06.930 --> 00:10:26.169
Ethan Jackson: Some very serious applied research in industries is based on techniques such as this. But it's a little bit out of reach for what we can do with constrained budgets, which is another justification for why we present the more simple version of things, where we just do kind of like a linear search.

63
00:10:26.200 --> 00:10:28.890
Ethan Jackson: over a hard coded structure.

64
00:10:30.840 --> 00:10:32.439
Ethan Jackson: Let's see here.

65
00:10:32.920 --> 00:10:38.819
Ethan Jackson: What I will say from this point is that these papers, again, provide

66
00:10:39.230 --> 00:10:44.220
Ethan Jackson: A lot of interesting ideas that we might be able to, to borrow from.

67
00:10:44.920 --> 00:10:52.520
Ethan Jackson: But we didn't implement any of them. And in fact, right when we were looking at the

68
00:10:52.540 --> 00:11:07.110
Ethan Jackson: actually, this just came out a couple of weeks ago when we were still working on the adaptive agent. There's a project from Microsoft called SkillOpt, that slots in very nicely, alongside what we've built. It's… it's…

69
00:11:07.110 --> 00:11:12.390
Ethan Jackson: basically this evolutionary process as presented in the other, papers.

70
00:11:12.390 --> 00:11:33.020
Ethan Jackson: but applied to the space of skills alone. And there's, you know, a lot of nice reasons. I kind of answered in the chat earlier why optimizing over the skills space alone has quite a lot of appeal. But even here, there's not this kind of like held out graduation gate that I think would make it very useful for forecasting in particular.

71
00:11:34.650 --> 00:11:55.930
Ethan Jackson: So let's take a look here at skills as a trainable state. This is what we're doing with the adaptive agent, but I just want to give you a flavor of what's going on in other tools like open source tools that are emerging so that you're aware. And if you want to experiment with these in the bootcamp, we can absolutely support you in doing that.

72
00:11:56.490 --> 00:12:13.449
Ethan Jackson: So the first is SkillOpt, and this is like an off-the-shelf package that's kind of ready to be experimented with. And the idea is very, very similar to what we have presented. The markdown skill files are a trainable state.

73
00:12:13.820 --> 00:12:22.460
Ethan Jackson: This is exactly what we did in the previous demo. Then there's an optimizer. This is a separate model that is going to

74
00:12:23.640 --> 00:12:39.979
Ethan Jackson: conduct a guided search over different, kind of, candidate ideas, for… for refinements to those skills. And then there's, like, they have a gate mechanism, it's just, you know, not specifically adapted for forecasting, that would… that would require some work.

75
00:12:40.190 --> 00:12:50.730
Ethan Jackson: But the point is that the edits are only accepted if it improves on a held-out task in Skillopt.

76
00:12:51.190 --> 00:13:05.270
Ethan Jackson: Sorry, held out evaluation criteria. So this is kind of much more aligned with like how you might want to optimize like a customer support bot or a coding agent, for example, not necessarily a forecasting agent.

77
00:13:08.070 --> 00:13:21.809
Ethan Jackson: I wanted to sneak this in. I actually don't know if this was included in the, material that we provided, but it was, like, there was this paper that we were reviewing in our discussion group just a couple of days ago, called SIA for self-approving agents.

78
00:13:21.810 --> 00:13:28.769
Ethan Jackson: And all I will say for this is that in addition to thinking of prompt and code.

79
00:13:28.770 --> 00:13:30.070
Ethan Jackson: as being, you know.

80
00:13:30.070 --> 00:13:41.569
Ethan Jackson: searchable during a process or subject of optimization. There are now techniques that are jointly optimizing the agent harness and the weights of the model that is powering that agent.

81
00:13:41.670 --> 00:14:01.519
Ethan Jackson: So it has both the text lever and the weight lever as knobs that can be tuned during the optimization process. And there's, you know, emerging work as of just like the last month or so that's showing that actually this can be very, very powerful. Something to keep in mind that agent optimization.

82
00:14:01.690 --> 00:14:18.160
Ethan Jackson: can touch any mutable part of the agent, which includes the prompts, the skills, the code around the LLM, and the LLM itself. These are all things that technically have plasticity, are mutable, and can be jointly optimized.

83
00:14:18.700 --> 00:14:22.550
Ethan Jackson: not saying that it's straightforward, just that it's it's Possible.

84
00:14:25.090 --> 00:14:30.490
Ethan Jackson: So again, the shared principle between what we've been talking about today.

85
00:14:30.790 --> 00:14:33.920
Ethan Jackson: is… that improvement

86
00:14:34.150 --> 00:14:52.229
Ethan Jackson: really should require a held-out date. With the Darwin-Godel machine, there's again this heavy emphasis on accepting or rejecting innovations based on withheld evaluations. With Alma, this really shifts it into the temporal domain.

87
00:14:52.340 --> 00:15:01.659
Ethan Jackson: where the quality of the memory and learning mechanism is judged specifically on this kind of like temporal, temporally cut off out of sample evaluation.

88
00:15:02.850 --> 00:15:07.159
Ethan Jackson: And one thing that we can borrow from SkillOpt is that,

89
00:15:07.160 --> 00:15:30.689
Ethan Jackson: We could very easily just bring this into our reference implementation rather than using an entire other package, but just edit the agent harness. If we built something that would commit a skill only if it improves validation on an unseen window, then we might have more confidence in the robustness of those or the quality of those edits. This is something that could be really, really simple, but we're

90
00:15:30.690 --> 00:15:32.629
Ethan Jackson: We're leaving that as an exercise.

91
00:15:34.630 --> 00:15:46.849
Ethan Jackson: The way that that could look is really just to add agent harness code that would propose a change to the strategy.

92
00:15:47.340 --> 00:15:55.919
Ethan Jackson: run held out evaluation in a more routine way, like we could build this actually into the harness so that it would be forced to rerun at least part of a back test.

93
00:15:56.050 --> 00:16:05.109
Ethan Jackson: And if the CRPS score is improved, then maybe we accept that. So the point being here that rather than kind of

94
00:16:05.810 --> 00:16:22.960
Ethan Jackson: Instructing the agent to run a back test. You can bring that out into the agent, into the harness that governs the update mechanism itself to say that like for this agent in this case, it has to run this test after it proposes to update its own strategy.

95
00:16:23.340 --> 00:16:27.709
Ethan Jackson: passes this improvement test, then we would commit.

96
00:16:28.190 --> 00:16:34.619
Ethan Jackson: So, I think that's kind of, like, the theme. If you're interested in exploring the self-improving agents track.

97
00:16:34.730 --> 00:16:43.000
Ethan Jackson: This is the direction that I would recommend, is adding this held-out gate to the agent harness, and this would be a very, kind of…

98
00:16:43.290 --> 00:16:58.499
Ethan Jackson: tractable way to explore this during the bootcamp. We only have a few days of building together, so I would encourage you to do something more like this than to go straight for the evolutionary search, which we don't have the budget for, realistically.

99
00:16:58.730 --> 00:17:01.909
Ethan Jackson: But I think we could do some meaningful exploration here.

100
00:17:02.170 --> 00:17:10.849
Ethan Jackson: So yeah, with that, I'll, I'll pause here, and I think we just have, like, I'll answer some questions in the chat, but I want to make sure that we're able to pass the floor to.

101
00:17:12.069 --> 00:17:20.529
Winnie Au: I think we have a quick minute to answer one question. So does it store the memory as embeddings or does it create tables or fields?

102
00:17:20.710 --> 00:17:38.190
Ethan Jackson: Good question. As far as I know, the memory, so it's actually open depending on how you frame the search space. You could constrain it either way, but for Alma, I believe that the memory mechanisms can be conditioned to use the file system.

103
00:17:38.190 --> 00:17:50.189
Ethan Jackson: if it wants to, but I see no reason why you couldn't configure it to… if, like, if it has access to, like, a vector embedding database in the environment, and tools to connect to that, it could use that. I think…

104
00:17:50.190 --> 00:17:54.690
Ethan Jackson: Yeah, the quick point that I'm seeing is, like, often memory…

105
00:17:54.690 --> 00:18:03.490
Ethan Jackson: using a RAG database can be overkill, and using the file system alone can be just as effective, if not more. But I'll stop there.

106
00:18:05.560 --> 00:18:12.309
Winnie Au: Sounds good. Thanks, Ethan. I'm going to quickly stop the recording and then restart it, Jesse, and then I'll hand it to you.
