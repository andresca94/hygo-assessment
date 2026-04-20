import {
  BadRequestException,
  Body,
  Controller,
  Get,
  HttpCode,
  Post,
  UploadedFile,
  UploadedFiles,
  UseInterceptors,
} from '@nestjs/common';
import { FileInterceptor, FilesInterceptor } from '@nestjs/platform-express';

import { CheckRequestDto } from '../dto/check-request.dto';
import { AgeSafetyService } from '../services/age-safety.service';
import { AgeSafetyBatchResult, AgeSafetyHealth, AgeSafetyResult } from '../types/age-safety.types';

const MAX_BATCH_FILES = Number(process.env.MAX_BATCH_FILES ?? 30);

@Controller('v1/age-safety')
export class AgeSafetyController {
  constructor(private readonly ageSafetyService: AgeSafetyService) {}

  @Get('health')
  async health(): Promise<AgeSafetyHealth> {
    return this.ageSafetyService.health();
  }

  @Post('check')
  @HttpCode(200)
  @UseInterceptors(FileInterceptor('file'))
  async checkImage(
    @UploadedFile() file: Express.Multer.File | undefined,
    @Body() metadata: CheckRequestDto,
  ): Promise<AgeSafetyResult> {
    if (!file) {
      throw new BadRequestException('A file field named "file" is required.');
    }
    return this.ageSafetyService.checkImage(file, metadata);
  }

  @Post('check-batch')
  @HttpCode(200)
  @UseInterceptors(FilesInterceptor('files', MAX_BATCH_FILES))
  async checkBatch(
    @UploadedFiles() files: Array<Express.Multer.File> | undefined,
    @Body() metadata: CheckRequestDto,
  ): Promise<AgeSafetyBatchResult> {
    if (!files || files.length === 0) {
      throw new BadRequestException('At least one file field named "files" is required.');
    }
    return this.ageSafetyService.checkBatch(files, metadata);
  }
}
