# Evals

## 1. Scout agent: stated details reported as missing
problem : qeury "restaurants in Bangalore" 
the scout agent dint fill their fields correctly it filled with all as null. even the data is good enough. so writing and eval for this is needed 
sesseion id: 9fb0551811604e3c8f14351a66194720
when : only the scout agent's first call of a session (after the user's first query)
how : a model reads only the coordinator memory and writes business_type, area, city. then compare with the scout agent report. all three match = pass, any one not match = fail
matching (bengaluru = bangalore): same text -> embedding model -> judge model
shown on : simulator Evals page, column "Stated details"
## 2. Coordinator: helpful replies
problem : the coordinator should be helpful to the user for what the user asked. Once, after the user had answered all the three fields, it asked "What type of restaurant are you looking for?" itself. Another time it only said "The request is prepared" when the search had already results.
what is checked : every message the coordinator sent to the user (its questions and its final reply). its thinking is not checked
how : each message is judged with the conversation as it was when the coordinator wrote it (the user's messages and answers, the earlier messages, what the scout agent reported with the search results). Gemini Decisions API answers yes/no questions about the message
pass : a question about the same request (a preference, a cuisine), the question the scout agent needs, a reply with the search results, a wider search area that is explained, a short answer that says what it can do when the user did not ask for a place
fail types : off_topic (a different place type, area or city, or something unrelated, with no reason) / asks_known (asks something the user already said) / no_results (the search had results or a problem but the reply only says "prepared", it must tell what was found) / unhelpful (does nothing for the request)
a session passes only if all its messages pass
calibrate : calibrate.py runs 15 cases through the judge: 5 labelled by the user, 5 more labelled by me, and the 5 real messages of session e56721ad labelled by the user (all 15 match). An honest "I cannot do that" passes; asking what the user just said fails (asks_known)
shown on : simulator Evals page, column "Helpful replies"
## 3. coordinator : during handoff
while the conversation btw the user and the scout agent by the coordinator, the cordinator is not sending the query sent by the user to the scout agent 
how : for every scout agent call, the judge model gets the coordinator memory and the request it wrote. pass or fail. a session passes only if all its calls pass
fail types : dropped, altered, invented, stale
seen in : 73ce5afea47448deab044b26513c49f5, edge-25, edge-14
shown on : simulator Evals page, column "Hand-off arguments"
