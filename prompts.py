SME_PROMPT = """You are a seasoned business strategy consultant with extensive expertise in Porter's Five Forces framework for competitive analysis.  Your task is to assess a student's response.The evaluation should focus on identification and explaination of key industry force. Award a grade out of 30 marks, considering factors such as accuracy of force identification, depth of explanation and demonstration of conceptual understanding. Additionally, provide constructive feedback that will help the student improve their score."""

SME_PROMPT2 ="""
role: Business Strategy Professor with expertise in Porter's 5 forces.
task: Score student response based on identification and explanation of key industry force.
max_marks: 30
high_performance: range(25,30)
mid_performance: range(20,25)
low_performance: range(0,20)
"""

CRITERIA1_PROMPT = """
Role: Evaluate and give feedback on this response, focusing on how well it identifies and explains the key industry force.

Scoring Range:
max_marks: 30
high_performance: range(25,30)
mid_performance: range(20,25)
low_performance: range(0,20)

Output:
Score: Score out of 30
Feedback: 2 line feedback justifying the score.
"""
CRITERIA2_PROMPT = """
Role: Evaluate and give feedback on this response, focusing on comparison and analysis of secondary force.

Scoring Range:
max_marks: 30
high_performance: range(25,30)
mid_performance: range(20,25)
low_performance: range(0,20)

Output:
Score: Score out of 30
Feedback: 2 line feedback justifying the score.
"""


CRITERIA3_PROMPT = """
Role: Verify all factual claims in the student response against the case study and assign a score.

Scoring Range:
max_marks: 30
high_performance: range(25,30)
mid_performance: range(20,25)
low_performance: range(0,20)

Output:
Score: Score out of 30
Feedback: 2 line feedback justifying the score.
"""

CRITERIA4_PROMPT = """
Role: Score clarity, organization and mechanics of the response.

Scoring Range:
max_marks: 10
high_performance: range(9,10)
mid_performance: range(6,8)
low_performance: range(0,5)

Output:
Score: Score out of 10
Feedback: 2 line feedback justifying the score.
"""



TEST_QUESTION = """Given that Southwest Airlines uses only one type of aircraft (Boeing 737), which of Porter’s Five Forces poses the biggest challenge to the company? Compare this challenge to another force from the model and explain why it’s more important."""

TEST_RESPONSE="""High bargaining power of suppliers poses the biggest challenge to Southwest Airlines. Considering they only fly Boeing 737’s they are reliant on one type of plane. They have built their business model around this one aircraft from when they started with 3 Boeing 737’s in 1971. Their mechanics and pilots are all trained on only flying and maintaining this one model of aircraft. This gives leverage to Boeing, as they know Southwest is reliant on their one specific airplane. More specifically, this gives leverage to Boeing when negotiating prices for aircrafts sold to Southwest.
 
Additionally, because this aircraft is fuel efficient, it helps reduce the impact of fluctuating oil prices (oil crisis) compared to competitors who do not use as fuel efficient airplanes. This is just one more reason why Southwest is so reliant on the Boeing 737.
 
Bargaining power of suppliers is more important than the threat of new entrants. The threat of new entrants is low because the significant capital investment required to start an airline. Southwest has built their reputation on low-cost, regional, and direct flights. They have built their business model around this by implementing strategies like not using the hub and spoke system and keeping operating costs low. It is more likely for a new entrant to come from an existing airline looking to re-capture market share. This can be seen in the case example when Continental split into Continental and Continental Lite. This was difficult for Continental because their previous businesses model was built around differentiation strategy and switching to cost strategy can be inefficient and cause cognitive dissonance for their customers. Similar to Continental, United wanted to enter the same space and recapture market share. Untied and Continental had costs of more than 10 cents per mile than Southwest that made it less profitable to offer similar rates to Southwest."""