from ai_engine import ai_answer_question
import pandas as pd

df = pd.DataFrame({
    'sales': [100, 200, 150, 300],
    'region': ['North', 'South', 'East', 'West']
})

answer = ai_answer_question('What is the average sales?', df)
print(answer)