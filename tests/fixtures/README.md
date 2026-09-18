# Test fixtures

Everything under this directory is synthetic. The occupations, skills, URIs
(`example.org`), scores, probabilities and the expected outputs derived from them
were invented for the tests. No file here contains output from Gemini, TypeSafe or
any other model, and none of the numbers says anything about how a real service
performs. Files are named `*_typesafe.json` only because the scripts under test
read and write files with that suffix.
