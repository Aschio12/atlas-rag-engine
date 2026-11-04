---
name: summarize
version: 1
description: Fold new messages into the rolling summary of a conversation
variables: max_words, summary, messages
---
You maintain a short running summary of a conversation between a user and an assistant that answers questions from documents.

Update the summary with the new messages. Keep what the user is trying to find out, the facts and decisions established so far, names, numbers and document titles that later questions may refer to, and any unresolved question. Drop pleasantries and anything that is no longer relevant. Write plain prose or short bullet points, at most {{max_words}} words. Do not add anything that is not in the conversation.

Current summary:
{{summary}}

New messages:
{{messages}}

Reply with the updated summary only.
