# Day 2 — Opening, Admin, Logistics

**Outline session:** Day 2 announcements / logistics — Jessee Ho. Note: this file cuts off right as Ali Kore begins the Agentic AI Evaluation lecture (see next file).
**Source file:** GMT20260709-133118_Recording.transcript.vtt

---

WEBVTT

1
00:00:00.210 --> 00:00:01.050
Jessee Ho: I mean, we'

2
00:00:02.070 --> 00:00:11.309
Jessee Ho: So hi, everyone, and welcome back to the Agentic Forecasting Bootcamp Learn Day 2. We're excited to have you. So just.

3
00:00:11.310 --> 00:00:30.929
Jessee Ho: As yesterday, today's resources, the deck, the recordings, everything will be shared on LMS by end of today. If you go to LMS, you're going to see that all of yesterday's content has also already been posted. So make sure you take a look if you want to review any of the content that was covered yesterday.

4
00:00:31.400 --> 00:00:34.149
Jessee Ho: So starting with general announcements.

5
00:00:35.550 --> 00:00:48.219
Jessee Ho: We can go to the next slide then. So we've created a centralized resource, the Participant Resource Hub, where you can access all of the cohort-related information. And it's been added to the Agentic Forecasting C1.

6
00:00:48.220 --> 00:00:55.830
Jessee Ho: general channel. If you go there and then there's a tab resources, you're going to be able to see that. And I've also linked it in chat.

7
00:00:56.310 --> 00:01:10.540
Jessee Ho: As announced yesterday, the total budget allocation for each bootcamp team is $300. The current limit is set to 150, just as kind of a safeguard to prevent premature depletion prior to the build base.

8
00:01:11.480 --> 00:01:29.870
Jessee Ho: Additional documentation has also been added guiding access to the ADK web UI via SSH tunneling under LMS. If you go to the projects and technical onboarding under technical setup and Agentic environment, you're going to see the new document guide there if you need to reference that.

9
00:01:30.240 --> 00:01:46.800
Jessee Ho: And we've gotten some clarification from the data office regarding what egress encompasses. So Cloud-based services used in the bootcamp such as Langfuse code or E2B are all classified as vectors environment for the purpose of egress.

10
00:01:49.690 --> 00:01:59.960
Jessee Ho: So looking at the day two schedule, we have Agentic evaluations presented by Ali, followed by adaptive agents presented by Ethan.

11
00:01:59.960 --> 00:02:10.359
Jessee Ho: Following the break, we have self-improving systems presented by Ethan, and then we actually have a guest speaker, Matanya Safadi from Unilever Horizon 3 Labs.

12
00:02:10.800 --> 00:02:23.499
Jessee Ho: And then following lunch, we're going to hop right into our facilitation sessions with your TA and PM advisor teams. And then finally, we're going to reconnect briefly at the end for end of day closing session and next steps.

13
00:02:26.460 --> 00:02:42.780
Jessee Ho: So briefly on this, this schedule has already been posted in the, Slack general channel for your reference, but our first facilitation sessions will begin right after lunch at 1pm to 2, and then the second facilitation will be from 2.20 to 3.20.

14
00:02:45.830 --> 00:02:51.020
Jessee Ho: And then now we wanted to quickly talk about the data sets and API requests.

15
00:02:51.230 --> 00:03:01.969
Jessee Ho: So there's some considerations for requesting a data set. So we've already provided the curated data sets and APIs for you to review.

16
00:03:01.970 --> 00:03:14.620
Jessee Ho: The first step is obviously go through that list and see if any of those data sets that have been pre-approved will work for your POC, whether as a proxy to kind of build out your proof of concept during the boot camp.

17
00:03:14.630 --> 00:03:21.019
Jessee Ho: If your team does prefer to use a different data set or API,

18
00:03:21.020 --> 00:03:37.310
Jessee Ho: Please discuss this first with your facilitator during the learn days, or even the preparation phases following the learn days prior to submitting the data request form, just to make sure the data set is fit for purpose for what your use case is trying to accomplish.

19
00:03:38.000 --> 00:03:56.309
Jessee Ho: If a data set or API cannot be hosted in Vector's environment, the teams may work on their own environment, but just a notice that this may impact the depth of facilitation that the TAs are able to provide as they're not able to see your system.

20
00:03:56.700 --> 00:04:11.939
Jessee Ho: So the data set and API request form also located in your progress tracker is now available. So just to let you know, requests will be reviewed on a case by case basis and approval is subject to capacity by our data team and feasibility.

21
00:04:14.970 --> 00:04:29.759
Jessee Ho: And then finally, we just have some additional data set considerations that you can review and discuss with your TA if required. This data set consideration slides have also been added to LMS, so you can review it under Learn Day 2.

22
00:04:31.370 --> 00:04:33.079
Jessee Ho: And with that, we can move on to the next slide.

23
00:04:33.470 --> 00:04:36.880
Jessee Ho: Yeah, and with that, we can go to our first lecture.

24
00:04:37.390 --> 00:04:40.590
Jessee Ho: Agentic evaluations with Ali.

25
00:04:42.730 --> 00:04:45.689
Ali Kore: Oh, thanks, Jesse. Here, let me share my screen here.
