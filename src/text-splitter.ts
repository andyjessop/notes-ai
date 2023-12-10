import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';

const splitter = new RecursiveCharacterTextSplitter({
  chunkOverlap: 200,
  chunkSize: 1536,
});

export async function splitFileIntoDocuments(content: string, fileName: string) {
  return splitter.createDocuments(
    [content],
    [
      {
        fileName,
        timestamp: Date.now(),
      },
    ],
    {
      appendChunkOverlapHeader: true,
      chunkHeader: `FILE NAME: ${fileName || 'None.'}\n\n---\n\n`,
    },
  );
}