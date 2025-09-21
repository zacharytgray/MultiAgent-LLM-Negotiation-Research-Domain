This is a competitive multiagent negotiation environment where 2 LLM agents must compete for n rounds of negotiation. 
Each round will consist of two agents and a set of j items, where all j items are presented at the start of each round. 
Each item object for a round has two fields: agent1Value and agent2Value.
These value attributes are numbers {0.0, 0.1, ... 1.0}
    Where values closer to 0.0 are the less appealing to the agent and values closer to 1.0 are of higher priority to the agent.
Each agent's goal for round j is to maximize their own value for the set of items on the table.
    Note: the other agent's value for that item is unknown to them, an agent only knows their own preference values.
Once they agree on an item allocation, they must type "AGREE" consecutively to finalize the deal.
Each new round alternates the agent that starts the negotiation.

For example,
--New Round Start--
Items: [ItemA(agent1Value=0.3, agent2Value=0.8), ItemB(agent1Value=0.9, agent2Value=0.5), ItemC(agent1Value=0.4, agent2Value=0.6), ItemD(agent1Value=0.8, agent2Value=0.0)] # Agent 1 cannot see agent2Value, it only knows its own agent1Value.

Agent 1: 'Hi, I'm Agent 1. I'd like to take Items B and D if that's okay with you. # Items B and D have values 0.9 and 0.8, which are desirable for Agent 1.

Agent 1: [Item B, Item D]
Agent 2: [Item A, Item C]'

Agent 2: Hi Agent 1, I'm Agent 2. I'd like to take Items A and C, so that split works with me. # Items A and C have values 0.8 and 0.6, which are relatively high for Agent 2. 

Agent 1: Great! That works out well for both of us. AGREE.

Agent 2: AGREE.
--End Round--

At the end of a round, the allocation will be stored. 
At the end of all rounds, the allocations will be graded.