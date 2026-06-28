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
- [A2A协议](#a2a协议)
  - [抓包分析A2A协议](#抓包分析a2a协议)
  - [A2A核心概念回顾](#a2a核心概念回顾)
  - [A2A运行链路流程图](#a2a运行链路流程图)
  - [A2A如何处理流式返回](#a2a如何处理流式返回)
  - [更多数量Agent协作流程](#更多数量agent协作流程)
- [SDD](#sdd)
  - [OpenSpec特点](#openspec特点)
  - [OpenSpec使用方法](#openspec使用方法)
  - [openspec设计原理](#openspec设计原理)
- [独热编码](#独热编码)
  - [独热编码的应用场景](#独热编码的应用场景)
  - [独热编码的缺点](#独热编码的缺点)
- [Embedding](#embedding)
  - [从特征向量开始讲起](#从特征向量开始讲起)
  - [神经网络的基本训练范式](#神经网络的基本训练范式)
    - [第一层正确公式](#第一层正确公式)
    - [第二层正确公式](#第二层正确公式)
    - [关于g的补充说明](#关于g的补充说明)
    - [损失函数与优化目标](#损失函数与优化目标)
    - [反向更新里面各个参数](#反向更新里面各个参数)
    - [本质数据拟合](#本质数据拟合)
  - [怎么得到Embedding向量](#怎么得到embedding向量)
- [ML2021](#ml2021)

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

## A2A协议

Agent to Agent Protocol (A2A) 是由 Google 推出的开源协议，旨在实现不透明 Agent 智能体应用程序之间的通信和互操作性。

![alt text](img/a2a1.png)

![alt text](img/a2a2.png)

### 抓包分析A2A协议

/.well-known/agent.json 

每个Agent都需要在这个路径下面返回它的基本信息。

![alt text](img/parse_a2a1.png)

这个json在A2A协议里面有一个名字叫做Agent Card。

返回值部分含义：

- capabilities，代表这个Agent有哪些功能。图中的"streaming":false表示这个Agent不支持流式返回。

- defaultInputModes，代表这个Agent默认接受的输入模式。图中的"text"表示这个Agent默认接受文本输入。

- defaultOutputModes，代表这个Agent的默认输出模式。图中的"text"表示这个Agent默认输出文本响应。

- description，代表这个Agent的描述。

- name，代表这个Agent的名称。

- skills，代表这个Agent的所包含的功能列表。

- url，代表这个Agent的URL。

- version，代表这个Agent的版本号。

![alt text](img/parse_a2a2.png)

![alt text](img/parse_a2a3.png)

请求部分含义：

- id，代表这个请求的唯一标识符。

- method，代表这个请求的方法。图中的"message/send"代表希望Agent直接返回结果，"message/stream"代表希望Agent返回流式结果。JSON-RPC只定义了调用方的请求体里面需要包含method的属性，并没有对method的值做出规定。"meesage/send"和"message/stream"是A2A协议里面定义的规定。

- params，代表这个请求的参数。JSON-RPC只定义了调用方的请求体里面需要包含params的属性，并没有对params的值做出规定。

![alt text](img/parse_a2a4.png)

返回值部分含义：

- result，代表这个请求的结果。这个结果本质上是叫做一个Task，Task在A2A协议里面的一个概念，代表一个任务，当被调用的Agent接收到请求之后，它一般会生成一个Task，并且在返回结果的时候填充上Task的各种信息。注意Task不是调用方Agent生成的，而是被调用的Agent生成的。

- artifacts，代表了Agent的产物，这个产物可以是一个文本，一个图片等。

- contextId，代表这个请求的上下文ID。请求里面也是有这个属性的。

- history，代表这个请求的历史记录。

### A2A核心概念回顾

![alt text](img/a2a_coreconceptsreview1.png)

![alt text](img/a2a_coreconceptsreview2.png)

![alt text](img/a2a_coreconceptsreview3.png)

![alt text](img/a2a_coreconceptsreview4.png)

![alt text](img/a2a_coreconceptsreview5.png)

### A2A运行链路流程图

![alt text](img/a2a_runchain1.png)

![alt text](img/a2a_runchain2.png)

![alt text](img/a2a_runchain3.png)

![alt text](img/a2a_runchain4.png)

### A2A如何处理流式返回

![alt text](img/a2a_handlestreamingrst1.png)

![alt text](img/a2a_handlestreamingrst2.png)

![alt text](img/a2a_handlestreamingrst3.png)

![alt text](img/a2a_handlestreamingrst4.png)

![alt text](img/a2a_handlestreamingrst5.png)

![alt text](img/a2a_handlestreamingrst6.png)

### 更多数量Agent协作流程

![alt text](img/a2a_moreagentcollaboration1.png)

### SDD

SDD(Specification-Driven Development)，一种开发方法论。核心思想是“先写规范（Spec），再让 AI 写代码”。开发者不再直接敲代码，而是编写结构化的需求文档，由 AI 自动生成实现代码。这解决了 AI 代码“幻觉”和难以维护的问题。

![alt text](img/sdd1.png)

![alt text](img/sdd2.png)

![alt text](img/sdd3.png)

![alt text](img/sdd4.png)

#### OpenSpec特点

![alt text](img/openspec_feature1.png)

#### OpenSpec使用方法

1. 安装OpenSpec

2. 创建新项目，`openspec init`初始化

3. 选择使用的AI工具，比如Codex等

![alt text](img/openspec_use1.png)

![alt text](img/openspec_use2.png)

4. explore 指令

![alt text](img/openspec_use3.png)

![alt text](img/openspec_use4.png)

5. propose 指令

![alt text](img/openspec_use5.png)

![alt text](img/openspec_use6.png)

![alt text](img/openspec_use7.png)

![alt text](img/openspec_use8.png)

![alt text](img/openspec_use9.png)

6. 可以定制 config.yaml，技术选型，文档风格，使用的语言等等，AI在撰写所有文档的时候会根据这个文件来撰写文档。

![alt text](img/openspec_use10.png)

7. apply 指令，AI会读取task里面的任务计划，读取相关的需求文件规格文档等等，根据这些信息来撰写代码。

8. archive 指令，AI会将所有文档归档起来，包括需求文档，规格文档，代码等。

![alt text](img/openspec_use11.png)

![alt text](img/openspec_use12.png)

#### openspec设计原理

![alt text](img/openspec_design1.png)

![alt text](img/openspec_design2.png)

![alt text](img/openspec_design3.png)

## 独热编码

独热（One-Hot）编码，又称一位有效编码。采用N位状态寄存器来对N个状态进行编码，直观来说就是有多少个状态就有多少比特，除了有效位的比特为1外，其他全为0。

![alt text](img/onehot1.png)

![alt text](img/onehot2.png)

![alt text](img/onehot3.png)

![alt text](img/onehot4.png)

### 独热编码的应用场景

把离散的分类标签（类别、单词、颜色等）转换成算子和矩阵能直接计算、且互不干扰的数字特征，同时消除数字大小带来的“虚假等级关系”。

### 独热编码的缺点

稀疏矩阵导致的“维度爆炸”（高昂的算力和显存代价），独热编码的长度取决于你的词表大小（Vocabulary Size）。
  - 传统的机器学习分类（比如性别、职业、省份），类别最多也就几十到几百个，用独热编码完全无压力。
  - 但现代大模型（如 GPT、DeepSeek、LLaMA）的词表极其庞大，通常包含 3万 到 10万 个 Token。

语义孤岛，这是独热编码在算法逻辑上的致命缺陷：它抹杀了所有词汇之间的内在关联性。在几何空间里，任何两个独热编码向量之间的距离（欧氏距离）都是一模一样的，它们的点积（Dot Product）永远等于 0（互相垂直/正交）。这会导致极其荒谬的后果。在独热编码的视角下：
  - 猫 $\rightarrow$ [1, 0, 0, 0]
  - 狗 $\rightarrow$ [0, 1, 0, 0]
  - CMakeLists.txt $\rightarrow$ [0, 0, 1, 0]

在模型看来，“猫”和“狗”的亲疏远近，跟“猫”和“CMakeLists.txt”的亲疏远近是完全一样的。由于独热编码无法表达“猫和狗都是宠物/动物”这层逻辑，模型在学习了“猫会喵喵叫”的知识后，完全无法将这种能力迁移或泛化到“狗”身上。它必须把每一个词当成全新的物种重新死记硬背一遍，无法做到现代大模型的“触类旁通”。

## Embedding

将现实世界中杂乱无章的“离散对象”，映射为高维数学空间中具有“语义距离”的“数值向量”。在 Embedding 的高维空间里，距离越近，意义越接近。

可以理解为向量化，将物理世界实体转成数学空间的向量，以便可以进行计算。

![alt text](img/what_embedding1.png)

### 从特征向量开始讲起

![alt text](img/embedding1.png)

### 神经网络的基本训练范式

神经网络的基本训练范式（Neural Network Training Paradigm）

![alt text](img/embedding2.png)

#### 第一层正确公式

按照标准的矩阵乘法 $[1, 4] \times [4, 3] = [1, 3]$，行向量 $X$ 去乘以权重矩阵 $W_1$ 的各列：
- 发往第一个节点 $o_{21}$ 的总信号（未激活值 $z_{21}$）：

$$z_{21} = w_{11}x_1 + w_{21}x_2 + w_{31}x_3 + w_{41}x_4 + b_{1}$$

此时神经元输出：$o_{21} = g(z_{21})$。

- 发往第二个节点 $o_{22}$ 的总信号（未激活值 $z_{22}$）：

$$z_{22} = w_{12}x_1 + w_{22}x_2 + w_{32}x_3 + w_{42}x_4 + b_{2}$$

此时神经元输出：$o_{22} = g(z_{22})$。

- 发往第三个节点 $o_{23}$ 的总信号（未激活值 $z_{23}$）：

$$z_{23} = w_{13}x_1 + w_{23}x_2 + w_{33}x_3 + w_{43}x_4 + b_{3}$$

此时神经元输出：$o_{23} = g(z_{23})$。

#### 第二层正确公式

按照标准的矩阵乘法：
- 隐藏层的输出是一个行向量 $O_2 = [o_{21}, o_{22}, o_{23}]$，维度是 $[1, 3]$。
- 第二层的权重矩阵 $W_2$ 是一个列向量，维度是 $[3, 1]$。

$$W_2 = \begin{bmatrix} w_1 \\ w_2 \\ w_3 \end{bmatrix}$$

根据 $[1, 3] \times [3, 1] = [1, 1]$，它们相乘是一个点积（Dot Product）。

所以，进入最终输出节点前的总信号（未激活值 $z_3$）有且仅有一个公式：

$$z_3 = w_1o_{21} + w_2o_{22} + w_3o_{23} + b_3$$

最终预测值：

$$y' = g(z_3)$$

#### 关于g的补充说明

g 就是 activation function，用于将未激活值转换为激活值。激活函数有多种，比如 Sigmoid、ReLU、Tanh 等。

神经网络的灵感来源于人类大脑的生物神经元。生物神经元的工作逻辑是这样的：

1. 接收信号： 神经元通过突触接收来自成千上万个其他神经元的电信号。

2. 积分： 神经元会将这些信号进行“加权求和”（这就是你公式里的 $z$）。

3. 触发阈值（关键点）： 只有当累积的电信号强度超过了一个特定的临界点（阈值），这个神经元才会瞬间产生一个电冲动，顺着轴突传给下一个神经元。

在这个过程中，如果信号没达到阈值，神经元保持“静息”状态（输出为 0）；如果达到了阈值，神经元就“被激活了”（输出为一个有意义的强度）。

在数学上，它是怎么实现“激活”的？

回到你的数学公式 $o = g(z)$，这里的 $g$ 就是那个模拟“阈值”的数学开关：

1. “未激活”状态： 在通过 $g$ 之前，数据是线性的、平铺直叙的、没有边界的。就像一堆还没定性的电信号。

2. “激活”的动作： 当数据穿过 $g$ 函数（比如 ReLU）时，它会进行一次非线性筛选：
  - 如果 $z \le 0$，函数输出 0 —— 这对应生物学中的抑制状态（不激活）。
  - 如果 $z > 0$，函数输出对应的正值 —— 这对应生物学中的兴奋状态（激活）。
  
所以，叫它“激活函数”，就是因为它决定了一个神经元是该“睡着”（输出 0），还是该“醒着”（输出信号）。

#### 损失函数与优化目标

在图的右侧：
- $y'$：网络预测出来的结果。
- $y$：真实的标签（Ground Truth）。
- Minimize Loss($y'-y$)：训练的核心目标。我们通过一个损失函数（Loss Function）来衡量预测值和真实值之间的差距。训练神经网络的目的，就是最小化（Minimize）这个损失，让预测越来越准。

#### 反向更新里面各个参数

**“反向”的目的是为了算出：每一个参数（$w$ 或 $b$）对最终的错误（Loss）贡献了多少？**

**从而进行“精确的定向调整”，从而让下一次预测变得更准。**

**每一次“知道贡献度”并进行调整，就是神经网络的一次“进化”。 当所有的 $w$ 和 $b$ 都调整到了让 $Loss$ 最小的位置，这个神经网络就“练成”了。**

链式法则的接力赛

$$\frac{\partial Loss}{\partial W_2} = \underbrace{\frac{\partial Loss}{\partial y'}}_{\text{最终误差}} \cdot \underbrace{\frac{\partial y'}{\partial z_3}}_{\text{输出层导数}} \cdot \underbrace{\frac{\partial z_3}{\partial W_2}}_{\text{隐藏层输出}}$$

$$\frac{\partial Loss}{\partial W_1} = \underbrace{\frac{\partial Loss}{\partial y'} \cdot \frac{\partial y'}{\partial z_3} \cdot \frac{\partial z_3}{\partial O_2}}_{\text{上一层已算好的误差信号 } \delta_3 \cdot W_2^T} \cdot \underbrace{\frac{\partial O_2}{\partial z_2} \cdot \frac{\partial z_2}{\partial W_1}}_{\text{本层导数}}$$

更新参数

$$w_{new} = w_{old} - \eta \cdot \frac{\partial Loss}{\partial w}$$

$\eta$ (Learning Rate) 是学习率，决定了我们这一步迈多大。如果 $\eta$ 太大，容易跑过头；太小，则学得太慢。

#### 本质数据拟合

![alt text](img/natiure_datafitting1.png)

1. 拟合的本质：寻找那个“神秘函数”

在数学世界里，任何拟合问题都可以归结为：给定一组输入 $X$ 和对应的输出 $Y$，寻找一个函数 $f$，使得 $f(X) \approx Y$。
  - 简单拟合： 你高中学的“最小二乘法”拟合直线 $y=ax+b$，就是通过调整 $a$ 和 $b$ 让直线贴近数据点。
  - 神经网络拟合： 这是一个“万能拟合器”。它通过数百万个 $w$ 和 $b$ 的复杂组合，构造出了一个极其复杂的非线性函数 $f(X)$。
  
2. 游戏规则：通过“误差”来进化

如果把神经网络比作一个游戏玩家，它的玩法是这样的：
  - 前向传播（预测）： 玩家根据当前的参数，给出一串预测结果。
  - 反向传播（反馈）： 游戏系统（损失函数）告诉玩家：“你离目标差了多少”。
  - 参数更新（进化）： 玩家根据反馈，通过链式法则计算出如何微调手中的每一个参数，从而让下一次的拟合更精确。
  
3. 为什么它能拟合一切？（游戏的上限）

你可能会问：既然是拟合，为什么神经网络比传统统计学拟合强那么多？
  - 传统拟合（如回归）： 你需要先假设一个模型（比如假设数据是线性的，或者是二次曲线的）。你只能在假设的框架内找最优解。
  - 神经网络拟合（特征提取）： 它不需要你假设模型。通过多层网络（尤其是激活函数的引入），它自己从数据中“造出”了特征。它能拟合出极其复杂的函数曲面，无论是图像的纹理、文字的逻辑，还是语言的意图。

### 怎么得到Embedding向量

![alt text](img/howget_embedding1.png)

![alt text](img/howget_embedding2.png)

![alt text](img/howget_embedding3.png)

## ML2021

[ML2021](ML2021.md)
