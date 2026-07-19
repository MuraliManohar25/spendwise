Spendwise is an AI-powered financial safety net for students managing money 
on their own for the first time — often away from home with no prior 
budgeting experience. Instead of just showing a current balance like a 
typical expense tracker, Spendwise analyzes daily spending patterns (trend, 
volatility, and pace) to predict whether a student is on track to overspend 
before month-end — and tells them exactly which category is driving the 
risk and what to change, before it becomes a crisis.

Most budgeting tools use simple linear math (spent-so-far ÷ days × 30). 
Spendwise instead trains a Random Forest classifier on spending *trajectory* 
features — is spending accelerating, is it spiky or steady, how does it 
compare to a fixed one-time cost like rent — because two students with the 
identical total spent so far can be headed toward very different outcomes 
depending on their pattern, something flat arithmetic can't catch.
