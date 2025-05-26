from rag_agent import load_agent, ask_question

agent = load_agent()

print("👋 歡迎使用 PBRT Q&A Agent，輸入你的問題吧！")

while True:
    q = input("你問：")
    if q.lower() in ["exit", "quit"]: break
    a = ask_question(agent, q)
    print("🤖 回答：", a)
