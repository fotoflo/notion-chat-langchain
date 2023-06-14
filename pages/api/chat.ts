import type { NextApiRequest, NextApiResponse } from 'next';
import { VectorDBQAChain } from 'langchain/chains';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { openai } from '@/utils/openai-client';
import { pinecone } from '@/utils/pinecone-client';
import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from '@/config/pinecone';

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  const { question } = req.body;

  if (!question) {
    return res.status(400).json({ message: 'No question in the request' });
  }

  try {
    // OpenAI recommends replacing newlines with spaces for best results
    const sanitizedQuestion = question.trim().replaceAll('\n', ' ');

    const prompt = `You are a helpful bot that answers questions about Accelerating Asia.
    The following is a conversation with a user. 
    If there's a link in the doc, please put it in an <a> tag and append it to https://www.notion.so/acceleratingasia/ 
    without the md extension.  Also who should i ask about, please include their contact info in the response
    The user is a human and you are a bot. \n\nUser: ${sanitizedQuestion}\nBot:`;

    const pineconeArgs = {
      pineconeIndex: pinecone.Index(PINECONE_INDEX_NAME),
      namespace: PINECONE_NAME_SPACE,
    };

    /* create vectorstore*/
    const vectorStore = await PineconeStore.fromExistingIndex(
      new OpenAIEmbeddings(),
      pineconeArgs,
    );

    const model = openai;
    // create the chain
    const chain = VectorDBQAChain.fromLLM(model, vectorStore);

    //Ask a question
    const response = await chain.call({
      query: prompt,
    });

    console.log('response', response);

    res.status(200).json(response);
  } catch (error: any) {
    console.log('error', error);
    res.status(500).json({ error: error?.message || 'Unknown error.' });
  }
}
