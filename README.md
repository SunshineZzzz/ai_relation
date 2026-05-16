- [MCP](#mcp)
  - [MCPClient](#mcpclient)
  - [MCPHost](#mcphost)
  - [MCPServer](#mcpserver)
    - [配置本地天气MCPServer](#配置本地天气mcpserver)
  - [MCPServer整个过程](#mcpserver整个过程)
  - [使用其他人的MCPServer](#使用其他人的mcpserver)
  - [MCP本质](#mcp本质)
  - [MCPHost与LLM之间的通信协议](#mcphost与llm之间的通信协议)
    - [告诉大模型环境信息](#告诉大模型环境信息)
    - [用户请求](#用户请求)
    - [大模型返回](#大模型返回)
    - [大模型请求MCP工具](#大模型请求mcp工具)
  - [Cline的XML协议与ReAct关系](#cline的xml协议与react关系)
- [Agent](#agent)
  - [Agent运行模式](#agent运行模式)
    - [ReAct模式](#react模式)
      - [ReAct是如何实现](#react是如何实现)
    - [PlanandExecute模式](#planandexecute模式)
    - [ReAct和PlanandExecute模式的区别](#react和planandexecute模式的区别)
- [Token到底是什么](#token到底是什么)
  - [Tokenizer](#tokenizer)
  - [BPE](#bpe)
- [AgentSkill](#agentskill)
- [ClaudeCode](#claudecode)
  - [终端使用](#终端使用)
  - [使用ClaudeSkill](#使用claudeskill)
    - [AgentSkillReference](#agentskillreference)
    - [AgentSkillScript](#agentskillscript)
    - [渐进式披露机制](#渐进式披露机制)
    - [Skill和Mcp的区别](#skill和mcp的区别)
- [RAG](#rag)
  - [RAG使用场景](#rag使用场景)
  - [RAG基本执行流程](#rag基本执行流程)
    - [RAG分片](#rag分片)
    - [RAG索引](#rag索引)
    - [向量数据库](#向量数据库)
    - [RAG召回](#rag召回)
    - [RAG重排](#rag重排)
    - [RAG生成](#rag生成)
  - [RAG总结](#rag总结)
- [ContextEngineering](#contextengineering)
  - [ContextWindow](#contextwindow)
  - [ContextEngineering兴起原因](#contextengineering兴起原因)
  - [ContextEngineering实现方法](#contextengineering实现方法)


## MCP

MCP(Model Context Protocol)，模型上下文协议，模型上下文协议的核心在于建立一个标准化的通信层，使得大型语言模型能够在处理用户请求或执行任务时，如果需要访问外部信息或功能，可以通过MCP客户端向MCP服务器发送请求。MCP服务器则负责与相应的外部数据源或工具进行交互，获取数据并按照MCP协议规范进行格式化，最后将格式化后的数据返回给大型语言模型。大型语言模型接收到上下文信息后，可以将其用于生成更准确、更相关的回复，或者执行用户请求的操作。

我的理解：

**MCP就是能够让大模型更好地使用各类工具的一个协议**

**MCP协议只规定了MCPClient与MCPServer之间的通信协议，不规定MCPHost与LLM之间的通信协议**

![alt text](img/mcp1.png)

### MCPClient

MCPClient是连接LLM和MCPServer的桥梁。负责：

1. 接收来自LLM的请求
2. 转换请求到MCP服务器
3. 将MCP服务器返回的结果返回给LLM

![alt text](img/mcpclient1.svg)

### MCPHost

MCPHost，本质就是支持MCP协议的软件，常见的有 Claude Desktop、Cursor、Cline，Cherry Studio 等。

1. VSC安装Cline插件
2. 配置Cline插件，可以使用OpenRouter，大模型聚合平台，你可以在这里一站式调用多个主流 AI 大模型，通过统一的 API 来体验和对比不同模型。你可以在自己开发网站的时候选择性接入任意大模型。

![alt text](img/mcphost1.svg)

### MCPServer

MCPServer，本地程序，提供工具和数据访问能力供 LLM 使用。与传统的远程 API 服务器不同，MCP 服务器可以作为本地应用运行在本地计算机上，也可以部署到远程服务器。

每个 MCPServer 都提供一组特定的工具，负责从本地数据或远程服务中检索信息。当 LLM（大语言模型） 确定在处理任务时需要使用某个工具时，它会通过 MCP 服务器提供的工具来获取所需数据，并将其返回给 LLM。

![alt text](img/mcpserver1.png)

![alt text](img/mcpserver2.png)

![alt text](img/mcpserver3.svg)

![alt text](img/mcpserver4.png)

#### 配置本地天气MCPServer

cline_mcp_settings.json
```JSON
{
  "mcpServers": {
    "weather-debug": {
      "command": "D:\\github\\ai_relation\\mcpserver\\weather\\main.exe",
      "timeout": 60,
      "transportType": "stdio",
      "autoApprove": [
        "get_weather"
      ]
    }
  }
}
```

```Go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

func main() {
	s := server.NewMCPServer(
		"Weather-MCPSvr",
		"1.0.0",
	)

	s.AddTool(mcp.NewTool("get_weather",
		mcp.WithDescription("Get the current weather conditions of the specified city."),
		mcp.WithString("city", mcp.Required(), mcp.Description("The name of the city to get the weather for.")),
	), func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		city, ok := request.GetArguments()["city"].(string)
		if !ok {
			return mcp.NewToolResultError("Invalid city parameter."), nil
		}

		fmt.Fprintf(os.Stderr, "Querying weather for city: %s\n", city)

		// 这里模拟调用 API，你可以换成真正的 http.Get 请求
		weatherInfo := fmt.Sprintf("The weather in %s is: Clear, Temperature 25°C", city)

		return mcp.NewToolResultText(weatherInfo), nil
	})

	// 启动基于 stdio 的服务器
	if err := server.ServeStdio(s); err != nil {
		fmt.Fprintf(os.Stderr, "Server error: %v\n", err)
	}
	// Test:
	// {"jsonrpc":"2.0","method":"tools/call","params":{"name":"get_weather","arguments":{"city":"Beijing"}},"id":1}
}
```

### MCPServer整个过程

![alt text](img/mcpserver_flow1.png)

![alt text](img/mcpserver_flow2.svg)

### 使用其他人的MCPServer

![alt text](img/mcpserver_useothers1.png)

```JSON
{
  "mcpServers": {
    "weather-debug": {
      "command": "D:\\github\\ai_relation\\mcpserver\\weather\\main.exe",
      "timeout": 60,
      "transportType": "stdio",
      "autoApprove": []
    },
    "mcp-server-hotnews": {
      "command": "npx",
      "args": [
        "-y",
        "@wopal/mcp-server-hotnews"
      ]
    }
  }
}
```

### MCP本质

![alt text](img/mcp_essence1.png)

![alt text](img/mcp_essence2.png)

![alt text](img/mcp_essence3.png)

![alt text](img/mcp_essence4.png)

![alt text](img/mcp_essence5.png)

### MCPHost与LLM之间的通信协议

![alt text](img/mcp_essence3.png)

可以搞一个中间服务器来将LLM和MCPHost之间的通信协议保存到文件中

![alt text](img/mcp_llm_protocol1.png)

现在主流模型聊天页面用的SSE

Server-Sent Events（SSE）是一种用于实现服务器向客户端实时推送数据的Web技术。与传统的轮询和长轮询相比，SSE提供了更高效和实时的数据推送机制。

SSE基于HTTP协议，允许服务器将数据以事件流（Event Stream）的形式发送给客户端。客户端通过建立持久的HTTP连接，并监听事件流，可以实时接收服务器推送的数据。

![alt text](img/mcp_llm_protocol2.png)

![alt text](img/mcp_llm_protocol3.png)

#### 告诉大模型环境信息

![alt text](img/mcp_llm_envinfo1.png)

![alt text](img/mcp_llm_envinfo2.png)

![alt text](img/mcp_llm_envinfo3.png)

![alt text](img/mcp_llm_envinfo4.png)

![alt text](img/mcp_llm_envinfo5.png)

#### 用户请求

用户请求 和 用户环境信息

![alt text](img/mcp_llm_userrequest1.png)

#### 大模型返回

冒号开头，其实就是注释返回，为了保持连接

后面就是流式返回

![alt text](img/mcp_llm_response1.png)

![alt text](img/mcp_llm_response2.png)

#### 大模型请求MCP工具

![alt text](img/mcp_llm_requesttool1.png)

MCP调用结果

![alt text](img/mcp_llm_requesttool2.png)

### Cline的XML协议与ReAct关系

ReAct 思想，是 Reason（推理） + Act（行动） 的缩写，是让大模型（LLM）学会“先思考，再行动”的一种逻辑框架。

一个典型的 ReAct 循环包含三个步骤：

1. Thought（思考）：AI 告诉自己：“我现在需要做什么？还缺什么信息？”

2. Action（行动）：AI 决定调用哪个工具（比如调用你的 Weather MCP）。    

3. Observation（观察）：AI 看到了工具返回的结果（比如“25°C”），然后根据这个结果进入下一个思考。

**ReAct 思想确实就是现代 AI Agent（智能体）的“灵魂”和运行逻辑。**

Agent，本就是**持续思考，持续调用外部工具，解决用户问题**

**Cline 本质上就是一个运行在 VS Code 里的全能型 AI Agent。**

![alt text](img/mcp_llm_react1.png)

ReAct思想和XML协议没有关系，可以使用任何协议，甚至是纯文本。

![alt text](img/mcp_llm_react2.png)

![alt text](img/mcp_llm_react3.png)

## Agent

大模型无法感知或者改变外界环境

![alt text](img/ai_agent1.png)

### Agent运行模式

#### ReAct模式

ReAct(Reasoning and Acting)，是一种让大模型（LLM）学会“先思考，再行动”的一种逻辑框架。

1. Thought（思考）：AI 告诉自己：“我现在需要做什么？还缺什么信息？”
2. Action（行动）：AI 决定调用哪个工具（比如调用你的 Weather MCP）。
3. Observation（观察）：AI 看到了工具返回的结果（比如“25°C”），然后根据这个结果进入下一个思考。
4. Final Answer（最终回答）：AI 综合所有思考和行动，给出最终的答案（比如“今天北京的天气是25°C，天气晴朗”）。

![alt text](img/ai_agent_react1.png)

##### ReAct是如何实现

系统提示词（System Prompt）

![alt text](img/ai_agent_howreact1.png)

![alt text](img/ai_agent_howreact2.png)

![alt text](img/ai_agent_howreact3.png)

[Agent代码](agent/agent.go)

![alt text](img/ai_agent_howreact4.png)

![alt text](img/ai_agent_howreact5.png)

#### PlanandExecute模式

![alt text](img/ai_agent_planandexecute1.png)

![alt text](img/ai_agent_planandexecute2.png)

![alt text](img/ai_agent_planandexecute3.png)

![alt text](img/ai_agent_planandexecute4.png)

![alt text](img/ai_agent_planandexecute5.png)

![alt text](img/ai_agent_planandexecute6.png)

![alt text](img/ai_agent_planandexecute7.png)

![alt text](img/ai_agent_planandexecute8.png)

#### ReAct和PlanandExecute模式的区别

| 维度 | PlanandExecute（规划-执行） | ReAct（动态推理） |
| ----- | ----- | ----- |
| 工作流 | 线性、静态：先一次性规划好所有步骤，再按顺序执行。 | 循环、动态：每一步都根据当前状态思考下一步做什么（Thought → Act → Observe）。 |
| 决策时机​ | 前期集中决策：规划完就不再改变计划。 | 实时动态决策：每一步都可能调整策略。
| 适用场景 | ​目标明确、步骤可预测的结构化任务（如写报告、数据分析）。| 环境多变、需要随机应变的探索性任务（如调试、复杂推理）。|
| 优点 | 计划清晰，易于调试和追踪。 | 灵活，能处理意外情况。|
| 缺点 | 死板：一旦计划出错或环境变化，容易全盘皆输。 | 冗余：每一步都要重新思考，Token 消耗大。|

## Token到底是什么

在 AI（尤其是大语言模型 LLM）领域，Token（切片/令牌） 是模型处理文本的基本单位。

![alt text](img/ai_agent_token1.png)

### Tokenizer

在大模型（LLM）的整个处理流程中，Tokenizer 是第一道也是最后一道关卡。你可以把它想象成一个“翻译官”，负责在 人类文字 和 模型能理解的数字（Token IDs） 之间进行转换。

Tokenizer（分词器） 是训练出来的

![alt text](img/ai_agent_tokenizer1.png)

### BPE

BPE（Byte Pair Encoding） 是一种常用的分词算法，用于将文本转换为模型可以理解的 Token IDs。

主流的算法（如 BPE - Byte Pair Encoding）的训练逻辑非常像“找共同点”：
1. 初始状态：把语料中所有的单个字符（a, b, c, 我, 他...）都看作初始词典。
2. 统计频率：在海量文本中寻找出现次数最多的“字符对”。
3. 合并：比如“t”和“h”经常一起出现，就合并成一个新的 Token “th”。
3. 循环往复：接着发现“th”和“e”经常一起出现，合并成“the”。
4. 停止：直到词典里的 Token 数量达到了预设的大小（比如 GPT-4o 的 20 万个，或者 Llama 的 3.2 万个）。

![alt text](img/ai_agent_bpe1.png)

![alt text](img/ai_agent_bpe2.png)

![alt text](img/ai_agent_bpe3.png)

![alt text](img/ai_agent_bpe4.png)

![alt text](img/ai_agent_bpe5.png)

## AgentSkill

最通俗的理解，AgentSkill其实就是一个大模型随时可以翻阅的说明文档

![alt text](img/ai_agent_skill1.png)

```TEXT
---
// 元数据(Metadata)
// AgentSkill名称
name：会议总结助手
// AgentSkill描述
description：该技能用于根据会议录音总结内容
---

// 下面都是指令(Instruction)
# 会议总结助手

## 总结规则

请将会议内容总结为以下几点：

- 参会人员
- 议题
- 决定

注意：每项都只能分别使用一句话来表述，不要分成多条。

// 实列

## 示例

输入：

张三：那我们开始吧，今天主要是把下个月社区志愿活动的安排一次性定下来。
李四：我建议活动放在公园，人多也方便组织。
王五：可以，不过要提前申请场地，不然可能有风险。
赵六：场地申请我可以负责，这周内给大家结果。
孙七：人数最好先有个范围，方便准备物资。
张三：那就先按50人左右来估算吧。
李四：上次的手套还能用，但垃圾袋需要再买。
王五：预算要不要设个上限，避免超支。
张三：预算控制在1000以内，优先用现有物资。
孙七：时间我建议周六上午，天气也不会太热。
李四：九点集合应该比较合适。
赵六：我周三前把申请结果同步到群里。
张三：好，那报名截止时间定在周四晚上。
王五：周五可以统一分组和采购。
孙七：我来负责写报名文案和活动当天的合影安排。
张三：安全方面提醒大家带水，活动结束简单总结一下就行。
张三：那今天就到这，大家按分工推进。

输出：

- 参会人员：张三、李四、王五、赵六、孙七
- 议题：统一确定下个月社区志愿活动的地点、时间、人数、预算及分工安排。
- 决定：活动定在公园并于周六上午九点举行，按约50人规模和1000预算执行，由赵六负责场地申请、孙七负责宣传及合影，其余成员配合物资和分组。
```

## ClaudeCode

### 终端使用

```CMD
PS C:\Windows\system32> irm https://claude.ai/install.ps1 | iex
Setting up Claude Code...

√ Claude Code successfully installed!

  Version: 2.1.22

  Location: C:\Users\Administrator\.local\bin\claude.exe


  Next: Run claude --help to get started

‼ Setup notes:
  • Native installation exists but C:\Users\Administrator\.local\bin is not in your PATH. Add it by opening: System
  Properties → Environment Variables → Edit User PATH → New → Add the path above. Then restart your terminal.


✅ Installation complete!
```

```CMD
@echo off
:: 设置字符集为 UTF-8 防止中文乱码
chcp 65001 >nul

echo 正在设置 Claude Code 环境变量以适配 Kimi 接口...

:: --- 在下面替换你真实的 API Key ---
set "MY_KEY="
:: ---------------------------------

:: 写入用户环境变量 (永久生效)
setx ANTHROPIC_BASE_URL "https://api.moonshot.cn/anthropic"
setx ANTHROPIC_AUTH_TOKEN "%MY_KEY%"
setx ANTHROPIC_MODEL "kimi-k2-turbo-preview"
setx ANTHROPIC_SMALL_FAST_MODEL "kimi-k2-turbo-preview"

echo.
echo ==================================================
echo [成功] 环境变量已写入 Windows 用户配置。
echo ==================================================

claude.exe
```

### 使用ClaudeSkill

![alt text](img/ai_agent_claudecode1.png)

![alt text](img/ai_agent_claudecode2.png)

![alt text](img/ai_agent_claudecode3.png)

![alt text](img/ai_agent_claudecode4.png)

#### AgentSkillReference

随着这个说明文档内容越来越多，浪费模型资源，所以我们需要一个AgentSkillReference，来按需加载

![alt text](img/ai_agent_skillreference1.png)

![alt text](img/ai_agent_skillreference2.png)

![alt text](img/ai_agent_skillreference3.png)

![alt text](img/ai_agent_skillreference4.png)

![alt text](img/ai_agent_skillreference5.png)

![alt text](img/ai_agent_skillreference6.png)

#### AgentSkillScript

**虽然Reference和Script都属于AgentSkill的高级功能，但是它们对模型上下文的影响其实是截然不同的，前者是读，会把内容加载到模型上下文里面，所以会消耗Token。后者是跑，它只会被执行，不会消耗Token。**

![alt text](img/ai_agent_skillscript1.png)

![alt text](img/ai_agent_skillscript2.png)

![alt text](img/ai_agent_skillscript3.png)

![alt text](img/ai_agent_skillscript4.png)

#### 渐进式披露机制

渐进式披露机制(Progressively disclose)

![alt text](img/ai_agent_progressivelydisclose1.png)

#### Skill和Mcp的区别

![alt text](img/ai_agent_skillmcp1.png)

## RAG

RAG(Retrieval-Augmented Generation)，是一种基于检索增强生成的技术，它通过将检索到的文档与用户输入的查询进行匹配，来生成更符合用户需求的响应。

![alt text](img/rag1.png)

### RAG使用场景

智能解决方案

![alt text](img/rag_intelligentsolution1.png)

![alt text](img/rag_intelligentsolution2.png)

### RAG基本执行流程

![alt text](img/rag_basicexecution1.png)

![alt text](img/rag_basicexecution2.png)

![alt text](img/rag_basicexecution3.png)

#### RAG分片

![alt text](img/rag_sharding1.png)

#### RAG索引

![alt text](img/rag_indexes1.png)

![alt text](img/rag_indexes2.png)

![alt text](img/rag_indexes3.png)

![alt text](img/rag_indexes4.png)

#### 向量数据库

![alt text](img/rag_vectordb1.png)

![alt text](img/rag_vectordb2.png)

**不管是分片还是索引，它们都发生在用户提问之前**

**下面是用户提问后的执行流程**

#### RAG召回

![alt text](img/rag_recall1.png)

![alt text](img/rag_recall2.png)

![alt text](img/rag_recall3.png)

#### RAG重排

![alt text](img/rag_rerank1.png)

![alt text](img/rag_rerank2.png)

![alt text](img/rag_rerank3.png)

#### RAG生成

![alt text](img/rag_generate1.png)

### RAG总结

![alt text](img/rag_summary1.png)

![alt text](img/rag_summary2.png)

## ContextEngineering

上下文工程(ContextEngineering)，**优化模型输入**，精心设计给模型的输入内容，让模型在有限的Context Window里面，尽可能理解的更准，答得更好，花的更少。核心思想，不改变模型的结构，只改变模型能"看到"的什么。

### ContextWindow

![alt text](img/contextwindow1.png)

![alt text](img/contextwindow2.png)

![alt text](img/contextwindow3.png)

![alt text](img/contextwindow4.png)

### ContextEngineering兴起原因

**Agent流行**

![alt text](img/contextengineering_risereason1.png)

### ContextEngineering实现方法

#### 保存Context

![alt text](img/contextengineeringimplementation_savecontext1.png)

#### 选择Context

![alt text](img/contextengineeringimplementation_selectcontext1.png)

#### 压缩Context

![alt text](img/contextengineeringimplementation_compresscontext1.png)

![alt text](img/contextengineeringimplementation_compresscontext2.png)

![alt text](img/contextengineeringimplementation_compresscontext3.png)

#### 隔离Context

![alt text](img/contextengineeringimplementation_isolation1.png)