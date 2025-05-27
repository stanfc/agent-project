from agent import load_agent, ask_question
import warnings

warnings.filterwarnings("ignore")
agent = load_agent()

print("🌍 Deep Research Online Agent 啟動，輸入你的問題吧！（輸入 exit 結束）")

while True:
    q = input("你問：")
    if q.lower() in ["exit", "quit"]: break
    a = ask_question(agent, q)
    print("🤖 回答：", a)
