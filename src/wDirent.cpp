#ifdef _WIN32
#include <iostream>
#include <string>
#include <stdio.h>
#include <windows.h>
#include "wDirent.h"

static HANDLE hFind;

DIR* opendir(const char* name)
{
	DIR* dir;
	WIN32_FIND_DATA FindData;
	char namebuf[512];

	//int sprintf ( char * str, const char * format, ... ); Write formatted data to string
	sprintf(namebuf, "%s\\*.*", name);

	hFind = FindFirstFile(namebuf, &FindData);
	if (hFind == INVALID_HANDLE_VALUE)
	{
		printf("FindFirstFile failed (%d)\n", GetLastError());
		return 0;
	}

	dir = (DIR*)malloc(sizeof(DIR));
	if (!dir)
	{
		printf("DIR memory allocate fail\n");
		return 0;
	}

	memset(dir, 0, sizeof(DIR));
	dir->dd_fd = 0;   // simulate return 

	return dir;
}

struct dirent* readdir(DIR* d)

{
	int i;

	BOOL bf;
	WIN32_FIND_DATA FileData;
	if (!d)
	{
		return 0;
	}

	bf = FindNextFile(hFind, &FileData);
	//fail or end 
	if (!bf)
	{
		return 0;
	}

	struct dirent* dir = (struct dirent*)malloc(sizeof(struct dirent) + sizeof(FileData.cFileName));

	for (i = 0; i < 256; i++)
	{
		dir->d_name[i] = FileData.cFileName[i];
		if (FileData.cFileName[i] == '\0') break;
	}
	dir->d_reclen = i;
	dir->d_reclen = FileData.nFileSizeLow;

	//check there is file or directory 
	if (FileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
	{
		dir->d_type = 2;
	}
	else
	{
		dir->d_type = 1;
	}

	return dir;
}

int closedir(DIR* d)
{
	if (!d)
		return -1;
	hFind = 0;
	free(d);
	return 0;
}
#endif
