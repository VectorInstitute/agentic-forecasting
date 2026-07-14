# Day 1 — Conventional Methods // S&P 500 Market Price Forecasting

**Outline session:** Overview of conventional methods // financial markets reference implementation (10:20–10:50am) — Behnoosh Zamanlooy
**Source file:** GMT20260708-142915_Recording.cutfile.20260708213130149.transcript.vtt

---

WEBVTT

1
00:00:00.000 --> 00:00:00.690
Behnoosh Zamanlooy (She/Her): Yeah, sure.

2
00:00:03.490 --> 00:00:11.520
Behnoosh Zamanlooy (She/Her): So hi, everyone. Like Ethan said, I'm going to just talk about a classic benchmarking and some conventional methods.

3
00:00:11.780 --> 00:00:17.740
Behnoosh Zamanlooy (She/Her): I'm going to talk about the S&P 500 modeling.

4
00:00:18.170 --> 00:00:35.969
Behnoosh Zamanlooy (She/Her): There are a lot of reasons why we're looking at the S&P 500. One, I think, kind of a compelling reason is that essentially modeling the S&P 500 and different indices of like.

5
00:00:36.170 --> 00:00:42.940
Behnoosh Zamanlooy (She/Her): The… basically, like, the… The prices of different, like, companies is

6
00:00:43.090 --> 00:00:45.349
Behnoosh Zamanlooy (She/Her): They, like, I don't know.

7
00:00:45.860 --> 00:00:56.679
Behnoosh Zamanlooy (She/Her): encourage the building of a subfield of mathematics called math finance. And there's a lot of research focused on it.

8
00:00:56.870 --> 00:01:11.149
Behnoosh Zamanlooy (She/Her): But and like a lot of people have their retirements invested in them and like basically a lot of big banks look into how they can forecast the variation in the returns of the S&P 500.

9
00:01:11.150 --> 00:01:21.840
Behnoosh Zamanlooy (She/Her): So what we're going to basically look at is like take the trend of the S&P 500, look at the returns at like different horizon next day, week and month.

10
00:01:21.840 --> 00:01:32.470
Behnoosh Zamanlooy (She/Her): And then look into, like, the measures of, deciding, let's say, whether you should hold S&P 500, whether you should sell, or not.

11
00:01:33.090 --> 00:01:47.280
Behnoosh Zamanlooy (She/Her): Now, what… like, a specific problem that we are going to look at is the overnight return for the S&P 500, and we can look at it at different horizons, so 1 business day, 5 business days, or 21 business days.

12
00:01:47.300 --> 00:01:56.319
Behnoosh Zamanlooy (She/Her): So for the one business day, it's going to be the, just, like, the overnight return, and for, like.

13
00:01:56.530 --> 00:02:05.540
Behnoosh Zamanlooy (She/Her): For anything bigger than that, we're going to look at the cumulative returns, basically. Like Ethan said.

14
00:02:05.660 --> 00:02:14.099
Behnoosh Zamanlooy (She/Her): we're going to look at multiple methods, so I'm going to focus on these, like, four first methods, which are conventional methods.

15
00:02:14.400 --> 00:02:23.400
Behnoosh Zamanlooy (She/Her): They are very good, and like Ethan says, sometimes hard to beat. So we have the naive method, which basically we're going to predict.

16
00:02:23.400 --> 00:02:39.490
Behnoosh Zamanlooy (She/Her): zero return at every step. We have the statistical methods like ARIMA, which we're going to look at, the past returns that we've seen, essentially a few lags, and the mistakes that we've made in forecasting.

17
00:02:39.490 --> 00:02:45.530
Behnoosh Zamanlooy (She/Her): There is methods like exponential smoothing and Kalman filter.

18
00:02:45.750 --> 00:02:56.240
Behnoosh Zamanlooy (She/Her): which I'm going to talk about a little bit, and then what I'm going to call machine learning. We're going to look at, gradient… tree gradient boosting methods, which are…

19
00:02:56.240 --> 00:03:14.520
Behnoosh Zamanlooy (She/Her): like one of the like Ethan said usually the top methods if you choose the covariates right and then the agents and the LLM processes that we're going to see later so just to give you an overview so the way so exponential smoothing works

20
00:03:14.520 --> 00:03:32.720
Behnoosh Zamanlooy (She/Her): basically, you just want to estimate two things, where the level of the market price is, and then, the past observed value that you had. And, you're going to make, essentially, the next forecast as this, like, let's say.

21
00:03:32.860 --> 00:03:47.919
Behnoosh Zamanlooy (She/Her): interpolation between the past level and the last observed value. You can also add, like, trends and seasonality to it, as well as covariates, if you want. All of this is possible, again, like Ethan said, in the DARTS package.

22
00:03:47.920 --> 00:04:04.490
Behnoosh Zamanlooy (She/Her): Another, like, kind of, like, very popular, univariate method is basically the Kalman filter. So, the Kalman filter has this idea within it that it's true that we observe the market prices.

23
00:04:04.490 --> 00:04:10.380
Behnoosh Zamanlooy (She/Her): But beneath it, there is always going to be like an unobservable state.

24
00:04:10.380 --> 00:04:21.160
Behnoosh Zamanlooy (She/Her): Think of it as a Kalman filter treats this unobservable hidden state as continuous. But let's say whether you're like in a bullish state, whether the market's steady or it's bearish.

25
00:04:21.160 --> 00:04:36.409
Behnoosh Zamanlooy (She/Her): And it tries to infer that state essentially from the past state that it has estimated, perhaps some covariate, and then the rest of it that it cannot explain is going to be kind of interpreted as white noise.

26
00:04:36.410 --> 00:04:44.659
Behnoosh Zamanlooy (She/Her): And the observation prediction is going to be basically a function of these.

27
00:04:44.950 --> 00:04:59.670
Behnoosh Zamanlooy (She/Her): these states that's estimated, plus some shocks and noise that the model cannot explain. Again, this is, like, parts of the DARS package and the way we have, like, our… like, our repo basically has wrapped, like, predictors around it.

28
00:04:59.780 --> 00:05:01.200
Behnoosh Zamanlooy (She/Her): Umm.

29
00:05:01.470 --> 00:05:06.489
Behnoosh Zamanlooy (She/Her): Another classical benchmark, again, autoregressive models.

30
00:05:06.870 --> 00:05:08.870
Behnoosh Zamanlooy (She/Her): So, they're just…

31
00:05:09.490 --> 00:05:17.140
Behnoosh Zamanlooy (She/Her): predicting the next level or the, you know, like the cumulative returns as a function of past returns.

32
00:05:17.180 --> 00:05:28.320
Behnoosh Zamanlooy (She/Her): Now, classically, what we used to do is, like, or you can still do it, and it's probably a good practice, is you look at the correlation between the market

33
00:05:28.320 --> 00:05:43.970
Behnoosh Zamanlooy (She/Her): let's say price now, and all of the possible lags, and this is going to tell you, basically, how you're going to choose the number of lags that you look at. We also look at partial autocorrelations, or partial correlations.

34
00:05:43.970 --> 00:05:46.749
Behnoosh Zamanlooy (She/Her): Which basically tells you, let's say, if…

35
00:05:46.750 --> 00:05:54.979
Behnoosh Zamanlooy (She/Her): I assume that I've already observed, let's say, lag one, then how much was the effect of the lag two?

36
00:05:54.980 --> 00:06:06.990
Behnoosh Zamanlooy (She/Her): which essentially takes care of, like, auto-correlated LAX. Now, DART, like, we're using auto-ARIMA, or auto-ARIMA methods. It takes care of this and estimates this.

37
00:06:06.990 --> 00:06:20.039
Behnoosh Zamanlooy (She/Her): But you can, of course, look at it. One good thing that I like about these methods is that they're pretty interpretable, so you give an understanding of how to go about modeling in the next steps.

38
00:06:20.040 --> 00:06:42.239
Behnoosh Zamanlooy (She/Her): We also have moving average methods, which basically treats the prediction as like, let's say, a certain level and then a function of the past mistakes that we've made in the prediction. Again, you can choose the number of past mistakes that we look at with correlation plots, but again, this is automatic.

39
00:06:42.240 --> 00:06:48.980
Behnoosh Zamanlooy (She/Her): So ARMA models or ARIMA models are Basically.

40
00:06:49.120 --> 00:06:53.539
Behnoosh Zamanlooy (She/Her): Just a combination of these 2 methods, we look at like.

41
00:06:53.910 --> 00:07:17.679
Behnoosh Zamanlooy (She/Her): Like we differences like ARIMA, the I comes from like basically the differences that we look at and modeling them. But since we're looking at the returns, the returns are steady, they're usually like close to stationary, so we don't need to do differencing. We don't cover it in the repo, but it's easy to do. You can always also add some covariates if you want to the models.

42
00:07:18.240 --> 00:07:40.920
Behnoosh Zamanlooy (She/Her): There are like a few covariates that we've included, but you can explore like further. So there are like volatility and indices, the rates of like let's say two years and ten years, some fed funds covariates, the unemployment rate returns, let's say on commodities, oil and gas and other like indices basically.

43
00:07:41.440 --> 00:07:58.060
Behnoosh Zamanlooy (She/Her): Now, these, we use them with, like, the gradient boosting… boosted trees. So the way, generally, even not considering trees, we look at gradient boosting is that we fit a first model, in this case tree, we look at… oops.

44
00:07:58.060 --> 00:08:10.019
Behnoosh Zamanlooy (She/Her): We look at where the… the points that we were not able to predict well, we look at the residuals, and we fit another model to it, and so on and so forth, basically.

45
00:08:10.220 --> 00:08:24.310
Behnoosh Zamanlooy (She/Her): Like Ethan said, the great thing about gradient boosting is that it works with mixed-stale covariates without having to do a lot of pre-processing, but, like, I will also show, like, you need to choose the covariates right.

46
00:08:24.330 --> 00:08:34.870
Behnoosh Zamanlooy (She/Her): So the way it works, so if I just mentioned the light GBM model as covariates, we've just passed like the five past lags of the returns.

47
00:08:34.870 --> 00:08:52.349
Behnoosh Zamanlooy (She/Her): And the LightGBM, the way it is implemented, it is like a depth-first tree. It just basically chooses based on each covariate what the cutoff should be, whether it is less than or more than.

48
00:08:52.620 --> 00:09:00.960
Behnoosh Zamanlooy (She/Her): And the leaf nodes, it is going to decide basically what the next forecast is going to be.

49
00:09:01.680 --> 00:09:11.539
Behnoosh Zamanlooy (She/Her): And where the gradient part of it is going to come from is that you're going to, like I said, fit the first tree and then

50
00:09:11.590 --> 00:09:16.899
Behnoosh Zamanlooy (She/Her): For the residuals of the first tree, a second one, and so on and so forth, basically.

51
00:09:16.930 --> 00:09:31.049
Behnoosh Zamanlooy (She/Her): The way it works is basically you're going to… the modeling playbook for this is you're going to choose the horizon, you're going to assemble the data, which is, like, streamlined in what I'm going to show you.

52
00:09:31.050 --> 00:09:44.360
Behnoosh Zamanlooy (She/Her): Create the features, so we have a bunch of features you can add to them if you want. Fit the baseline, fit the models that have covariates, and then do backtesting. So for these models, backtesting is…

53
00:09:44.360 --> 00:09:51.960
Behnoosh Zamanlooy (She/Her): easy because we can make sure that the cutoff is made so we don't have the problems that we might have with LLM processes.

54
00:09:52.230 --> 00:10:04.619
Behnoosh Zamanlooy (She/Her): What we're going to look at as a measure of success, let's say, for the classical models, like Ethan said, as CRPS, so essentially.

55
00:10:04.620 --> 00:10:14.269
Behnoosh Zamanlooy (She/Her): Just CRPS tells you how widely spread is your forecast. So we make multiple forecasts.

56
00:10:14.270 --> 00:10:27.659
Behnoosh Zamanlooy (She/Her): Using Monte Carlo methods and how widely spread it is around like the actual price point, the less spread it is, the better for us. It means like basically the model is sure.

57
00:10:28.080 --> 00:10:39.099
Behnoosh Zamanlooy (She/Her): more sure about the prediction that it makes, and empirically, the way it works is that you have these, like, true, value, and you make the empirical CDF

58
00:10:39.100 --> 00:10:49.879
Behnoosh Zamanlooy (She/Her): based on the forecast that you've made and then you look at essentially this integration part so you can think of it as the distance between like a point

59
00:10:49.880 --> 00:10:53.289
Behnoosh Zamanlooy (She/Her): point distribution and the Cdf that you have.

60
00:10:53.870 --> 00:11:02.679
Behnoosh Zamanlooy (She/Her): Now, there are multiple things that you can look at, multiple periods that you could look at.

61
00:11:02.680 --> 00:11:21.630
Behnoosh Zamanlooy (She/Her): A specific one that we have looked at is the period of COVID-19 where there was a lot of uncertainty and it's a harder period to predict usually for the classical models. So we've taken a year before this period for training and then we make the forecast.

62
00:11:21.780 --> 00:11:24.129
Behnoosh Zamanlooy (She/Her): So, yeah.

63
00:11:24.350 --> 00:11:33.299
Behnoosh Zamanlooy (She/Her): What you can see, and it's kind of expected, is that the covariates are usually good for one business day ahead, but

64
00:11:34.250 --> 00:11:42.929
Behnoosh Zamanlooy (She/Her): when it comes to multiple, like, horizons, it doesn't perform as well. Like I said, like, it's…

65
00:11:43.390 --> 00:11:51.509
Behnoosh Zamanlooy (She/Her): like, some of the classical models are actually relatively good, like AutoArena, you can see it is performing

66
00:11:51.510 --> 00:12:04.329
Behnoosh Zamanlooy (She/Her): like, closer to the light GBM, so they're not something to discard. Classically, what you would do is, like, maybe bagging of these models, but I think that's something that the LLMs can be very good at.

67
00:12:04.330 --> 00:12:08.579
Behnoosh Zamanlooy (She/Her): One more thing that I would say is that you.

68
00:12:08.580 --> 00:12:27.900
Behnoosh Zamanlooy (She/Her): classically you have to look at maybe multiple measures. The CRPS is going to be our main measure that we look at but if we look at how well just the returns going up or down is forecasted for the same period.

69
00:12:27.900 --> 00:12:35.529
Behnoosh Zamanlooy (She/Her): So I'm looking at the area under the curve of basically classifying the returns going up or down.

70
00:12:35.530 --> 00:12:47.329
Behnoosh Zamanlooy (She/Her): What you're going to see is that the models with covariates don't perform well. So if you're below 0.5, you're even performing worse than random guessing.

71
00:12:47.330 --> 00:12:57.950
Behnoosh Zamanlooy (She/Her): In these cases, you can see that the classical models perform better. One thing that I said is basically the covariates that maybe we've chosen too many covariates, the covariates.

72
00:12:58.050 --> 00:13:06.130
Behnoosh Zamanlooy (She/Her): Perhaps like they capture some of the shocks, to the market, but they're not, good at steady periods.

73
00:13:06.680 --> 00:13:07.820
Behnoosh Zamanlooy (She/Her): So…

74
00:13:08.310 --> 00:13:21.590
Behnoosh Zamanlooy (She/Her): I'm now going to kind of quickly go over this script that we have in our repo that basically I've used to produce all these results.

75
00:13:21.820 --> 00:13:26.199
Behnoosh Zamanlooy (She/Her): Let me just quickly share it.

76
00:13:26.470 --> 00:13:30.960
Behnoosh Zamanlooy (She/Her): yes.

77
00:13:31.990 --> 00:13:38.849
Behnoosh Zamanlooy (She/Her): here we go. Okay, so this is going to be under the Snp. 500 forecasting.

78
00:13:39.020 --> 00:13:55.929
Behnoosh Zamanlooy (She/Her): and this notebook. So here is just mostly an explanation of what I went through, the models that we are covering, the ones that you can pass covariates to in our notebook.

79
00:13:56.680 --> 00:14:01.299
Behnoosh Zamanlooy (She/Her): And how, basically, for the classical methods, it is…

80
00:14:01.360 --> 00:14:04.519
Behnoosh Zamanlooy (She/Her): Easy to do cutoff aware evaluation.

81
00:14:04.520 --> 00:14:26.609
Behnoosh Zamanlooy (She/Her): But once, like when you are like in the 2020 year, it is hard to force, let's say, LLMP processes. So just using LLMs simply as forecasting, nothing agentic, just passing the data that other models see and then asking it to make a prediction, we're going to see leakage essentially.

82
00:14:28.010 --> 00:14:37.599
Behnoosh Zamanlooy (She/Her): So what essentially you have to do, this is just going to be loading the data that you need.

83
00:14:37.790 --> 00:14:47.910
Behnoosh Zamanlooy (She/Her): So, for the forecasting problems, you have specifications, that you're going to make, so… And…

84
00:14:47.910 --> 00:15:08.549
Behnoosh Zamanlooy (She/Her): specifications that you have to make. So this is going to be essentially the specifics about your projection methods. I'm going to, let's say, like the smoke test that is in this in this example. So you essentially are going to have like a task id, which is always going to be

85
00:15:08.550 --> 00:15:13.389
Behnoosh Zamanlooy (She/Her): this for this problem and then the horizons that you want to

86
00:15:13.390 --> 00:15:21.529
Behnoosh Zamanlooy (She/Her): make predictions about. So 1, 5, and 21 and then the start and end of the

87
00:15:22.300 --> 00:15:30.100
Behnoosh Zamanlooy (She/Her): The predictions you are going to make and the number of points before the study you're going to use to fit the model essentially.

88
00:15:31.900 --> 00:15:35.389
Behnoosh Zamanlooy (She/Her): And here in the

89
00:15:35.390 --> 00:15:52.149
Behnoosh Zamanlooy (She/Her): Let me see. In the predictions, what is going to happen is that you're going to save the prediction each time you run the notebook. It takes a long time to fit the baseline models, maybe around like two hours if you do the COVID-19 problem. So that's nice.

90
00:15:52.150 --> 00:16:06.710
Behnoosh Zamanlooy (She/Her): Then, here in the configuration, you can choose which of the three ones that we have provided, which experiment you're going to run. So if this is smoke, it's going to correspond to the specification that it showed.

91
00:16:07.800 --> 00:16:16.859
Behnoosh Zamanlooy (She/Her): The second… Yep. And like I said, this is just going to be a

92
00:16:17.290 --> 00:16:32.460
Behnoosh Zamanlooy (She/Her): like fitting up the baseline models. The lags is going to be the number of lags that's passed as covariate to baseline Gbm. And the number of sample is going to be the number of samples that is used that is generated using the Monte Carlo method just from the

93
00:16:32.460 --> 00:16:39.610
Behnoosh Zamanlooy (She/Her): yards package, and then the arguments that we need for our gradient tree boosting methods.

94
00:16:40.380 --> 00:16:41.820
Behnoosh Zamanlooy (She/Her): I don't know.

95
00:16:43.110 --> 00:16:58.769
Behnoosh Zamanlooy (She/Her): And finally, you're going to have a leaderboard, so the leaderboard is going to show you the, CRPS, results, or lower is better, and then you're going to have a bunch of, other

96
00:16:58.860 --> 00:17:16.870
Behnoosh Zamanlooy (She/Her): other methods like the directional accuracy the area under the curve of how well we've just predicted up and down but this is mostly I would say like meaningful for the next day. You can also use it for the other horizons.

97
00:17:18.530 --> 00:17:28.459
Behnoosh Zamanlooy (She/Her): And then, like, graphing of the best methods, basically, versus the naive methods, if you want to take a closer look. And…

98
00:17:29.080 --> 00:17:36.590
Behnoosh Zamanlooy (She/Her): This is just, going to be related, to, like, I don't know

99
00:17:36.890 --> 00:17:44.320
Behnoosh Zamanlooy (She/Her): to, like, building, like, a specific leaderboard for those methods. Like, the comparison of maybe two methods.

100
00:17:46.550 --> 00:17:53.869
Behnoosh Zamanlooy (She/Her): This is, I think, a summary of the baseline methods. If you have any questions, I'm happy to answer them.

101
00:18:08.050 --> 00:18:09.353
Ethan Jackson: Great. Thank you, Manoj. I think I might pass it back to Winnie. We're nearing our first break time.
