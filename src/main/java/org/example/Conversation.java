package org.example;

import java.util.ArrayList;
import java.util.List;
import dev.langchain4j.model.chat.*;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;

/**
 * A class to represent a conversation with a chatbot.
 * It hides the details of LangChain4j and OpenAI, so client can
 * ask just call 'askQuestion' with context (instructions, etc.) and
 * the question, and get a string back. The conversation thus far is sent
 * to OpenAI on each question.
 *
 * Client can also get sample questions based on a given context, and can reset
 * conversation to start over.
 */
public class Conversation {
    private MessageWindowChatMemory chatMemory;
    private ChatLanguageModel chatModel;

    // Constructor
    public Conversation(String apiKey) {
        this.chatModel = OpenAiChatModel.withApiKey(apiKey);
        this.chatMemory=MessageWindowChatMemory.withMaxMessages(10);
    }

    /** askQuestion allows user to ask a question with context (e.g., instructions
     * for how OpenAI should respondd. It adds the context and question to the memory,
     * in the form langchain4j wants, then asks the question, then puts response into memory and returns text of response. in the form
     */

    public String askQuestion(String context, String question) {
        SystemMessage sysMessage = SystemMessage.from(context);
        chatMemory.add(sysMessage);
        UserMessage userMessage = UserMessage.from(question);
        chatMemory.add(userMessage);
        // Generate the response from the model
        Response response = chatModel.generate(chatMemory.messages());
        AiMessage aiMessage = (AiMessage) response.content();
        chatMemory.add(aiMessage);
        String responseText = aiMessage.text();

        return responseText;
    }

    /**
     * Method to generate sample questions with a given context. You can specify the number of questions
     * and the max words that should be generated for each question. This method is
     * often used to provide user with sample questions to trigger the dialogue.
     */
    public List<String> generateSampleQuestions(String context, int count, int maxWords) {
        List<String> questions = new ArrayList<>();
        String instructions = "For the context following, please provide a list of " + count + " questions with a maximum of " + maxWords + " words per question.";
        instructions = instructions + " Return the questions as a string with delimiter '%%' between each generated question";
        SystemMessage sysMessage = SystemMessage.from(instructions );
        UserMessage userMessage = UserMessage.from(context);
        List<ChatMessage> prompt = new ArrayList<>();
        prompt.add(sysMessage);
        prompt.add(userMessage);
        Response response = chatModel.generate(prompt);
        AiMessage aiMessage = (AiMessage) response.content();
        String responseText = aiMessage.text();
        String[] questionArray = responseText.split("%%");
        return List.of(questionArray);
    }


    public void resetConversation() {

        chatMemory.clear();
    }

    /**
     *
     * @return
     */
    public String toString() {

        return chatMemory.messages().toString();
    }
    /**
     *
     * main is a sample that asks two questions, the second of which
     * can only be answered if model remembers the first.
     */

    public static void main(String[] args) {
        // Example conversation
        Conversation conversation = new Conversation("demo");
        // Generate sample questions
        List<String> questions = conversation.generateSampleQuestions("Questions about films in the 1960s", 3, 10);
        System.out.println("Sample questions: " + questions);

        // Ask a question
        String response = conversation.askQuestion("You are a film expert", "What are the three best Quentin Tarintino movies?");
        System.out.println("Response: " + response);

        // Ask another question to show continuation-- openAI knows 'he' is Tarantino from memory
        response = conversation.askQuestion("You are a film expert", "How old is he");
        System.out.println("Response: " + response);

        // Print conversation history
        System.out.println("\nConversation History:");
        System.out.println(conversation);

    }

}
