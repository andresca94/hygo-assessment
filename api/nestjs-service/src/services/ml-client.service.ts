import { Injectable, ServiceUnavailableException } from '@nestjs/common';
import axios, { AxiosInstance } from 'axios';
import FormData from 'form-data';

import { CheckRequestDto } from '../dto/check-request.dto';
import { AgeSafetyBatchResult, AgeSafetyHealth, AgeSafetyResult } from '../types/age-safety.types';

@Injectable()
export class MlClientService {
  private readonly client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: process.env.ML_INFERENCE_URL ?? 'http://localhost:8000',
      timeout: 120000,
    });
  }

  async health(): Promise<AgeSafetyHealth> {
    try {
      const response = await this.client.get<AgeSafetyHealth>('/health');
      return response.data;
    } catch (error) {
      throw new ServiceUnavailableException('Inference service is not reachable.');
    }
  }

  async checkImage(file: Express.Multer.File, metadata: CheckRequestDto): Promise<AgeSafetyResult> {
    const form = new FormData();
    form.append('file', file.buffer, {
      contentType: file.mimetype,
      filename: file.originalname,
    });
    if (metadata.requestId) {
      form.append('requestId', metadata.requestId);
    }
    if (metadata.source) {
      form.append('source', metadata.source);
    }

    const response = await this.client.post<AgeSafetyResult>('/v1/age-safety/check', form, {
      headers: form.getHeaders(),
      maxBodyLength: Infinity,
    });
    return response.data;
  }

  async checkBatch(files: Express.Multer.File[], metadata: CheckRequestDto): Promise<AgeSafetyBatchResult> {
    const form = new FormData();
    for (const file of files) {
      form.append('files', file.buffer, {
        contentType: file.mimetype,
        filename: file.originalname,
      });
    }
    if (metadata.requestId) {
      form.append('requestId', metadata.requestId);
    }
    if (metadata.source) {
      form.append('source', metadata.source);
    }

    const response = await this.client.post<AgeSafetyBatchResult>('/v1/age-safety/check-batch', form, {
      headers: form.getHeaders(),
      maxBodyLength: Infinity,
    });
    return response.data;
  }
}
