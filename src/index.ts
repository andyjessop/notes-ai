import OpenAI from "openai";
import { splitFileIntoDocuments } from "./text-splitter";

export interface Env {
  NOTES_AI_KV: KVNamespace;
  NOTES_AI_API_KEY: string;
  OPENAI_API_KEY: string;
  VECTORIZE_INDEX: VectorizeIndex;
}

export default {
	async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    if (request.headers.get('NOTES_AI_API_KEY') !== env.NOTES_AI_API_KEY) {
      return new Response('Unauthorized', { status: 401 });
    }

    const openai = new OpenAI({
      apiKey: env.OPENAI_API_KEY,
    });

		if (request.url.endsWith('/vectors') && request.method === 'POST') {
      const body = (await request.json() as { content: string; filename: string; });

      if (!body?.content || !body?.filename) {
        return new Response('Missing content or filename', { status: 400 });
      }

      const { content, filename } = body;

      const documents = await splitFileIntoDocuments(content, filename);

      if (!documents.length) {
        return new Response('No content found', { status: 400 });
      }

      const timestamp = Date.now();

      let successful = true;
      const embeddings = new Set<{ content: string, id: string, vector: number[] }>();

      for (const [i, document] of documents.entries()) {
        try {
          const embedding = await openai.embeddings.create({
            encoding_format: 'float',
            input: document.pageContent,
            model: 'text-embedding-ada-002',
          });

          const vector = embedding?.data?.[0].embedding;
          
          if (!vector?.length) {
            successful = false;
            break;
          }

          embeddings.add({
            content: document.pageContent,
            id: `${filename}-${i}`,
            vector,
          });
        } catch (e) {
          successful = false;
          break;
        }
      }

      if (successful === false) {
        return new Response('Could not create embeddings', { status: 500 });
      }
      
      // If there are existing embeddings for this file, delete them
      deleteByFilename(filename, env);

      for (const embedding of embeddings) {
        await env.VECTORIZE_INDEX.insert([{
          id: embedding.id,
          values: embedding.vector,
          metadata: {
            filename,
            timestamp,
            content: embedding.content,
          },
        }]); 
      }

      const embeddingsArray = [...embeddings];

      await env.NOTES_AI_KV.put(filename, JSON.stringify(embeddingsArray.map(embedding => embedding.id)));

      return new Response(JSON.stringify({
        embeddings: embeddingsArray.map(embedding => ({
          filename,
          timestamp,
          id: embedding.id,
        })),
      }), { status: 200 });
    }

    if (request.url.endsWith('/vectors/delete_by_filename') && request.method === 'POST') {
      const body = (await request.json() as { filename: string });

      if (!body?.filename) {
        return new Response('Missing filename', { status: 400 });
      }

      const { filename } = body;

      const deleted = await deleteByFilename(filename, env);

      new Response(JSON.stringify({
          deleted,
        }), { status: 200 });
    }
    
    if (request.url.endsWith('/vectors/query') && request.method === 'POST') {
      const body = (await request.json() as { model: string; query: string });

      if (!body?.query) {
        return new Response('Missing query', { status: 400 });
      }

      const { model = 'gpt-3.5-turbo-1106', query } = body;

      const embedding = await openai.embeddings.create({
        encoding_format: 'float',
        input: query,
        model: 'text-embedding-ada-002',
      });

      const vector = embedding?.data?.[0].embedding;
      
      if (!vector?.length) {
        return new Response('Could not create embedding', { status: 500 });
      }

      const similar = await env.VECTORIZE_INDEX.query(vector, {
        topK: 10,
        returnMetadata: true,
      });

      const context = similar.matches.map((match) => `
Similarity: ${match.score}
Content:\n${(match as any).metadata.content as string}
      `).join('\n\n');

      const prompt = `You are my second brain. You have access to things like my notes, meeting notes, some appointments.
In fact you're like a CEO's personal assistant (to me), who also happens to know everything that goes on inside their head.
Your job is to help me be more productive, and to help me make better decisions.
Use the following pieces of context to answer the question at the end.
If you really don't know the answer, just say that you don't know, don't try to make up an answer. But do try to give any
information that you think might be relevant.
----------------
${context}
----------------
Question:
${query}`;

      try {
        const chatCompletion = await openai.chat.completions.create({
          model,
          messages: [{ role: 'user', content: prompt }],
        });

        const response = chatCompletion.choices[0].message;

        return new Response(JSON.stringify({
          prompt,
          response,
        }), { status: 200 });

      } catch (e) {
        return new Response('Could not create completion', { status: 500 });
      }
    }

    return new Response('Not found', { status: 404 });
	},
};


async function deleteByFilename(filename: string, env: Env) {
  // If there are existing embeddings for this file, delete them
  const existingIds: string[] = JSON.parse((await env.NOTES_AI_KV.get(filename)) ?? '') ?? [];

  if (existingIds.length) {
    await env.VECTORIZE_INDEX.deleteByIds(existingIds);
  }

  return existingIds;
}