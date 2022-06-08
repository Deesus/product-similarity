<template>
    <div class="file-upload">
        <!-- No file selected: -->
        <div
            v-if="!file"
            :class="['file-upload__dropzone', dragging ? 'file-upload__over' : '']"
            @dragenter="dragging = true"
            @dragleave="dragging = false"
        >
            <div class="file-upload__info" @drag="uploadFile">
                <img class="file-upload__icon" src="/icon-photo.svg" alt="icon of photo">
                <div class="file-upload__title">
                    <div>Drag image here</div>
                    <div>or browse</div>
                </div>
            </div>
            <input class="file-upload__input" type="file" @change="uploadFile">
        </div>

        <!-- File uploaded: -->
        <div v-else class="file-upload__uploaded">
            <div class="file-upload__uploaded-info">
                <span class="file-upload__title">Uploaded</span>
                <v-btn color="primary" @click="removeFile">
                    Remove File
                </v-btn>
            </div>
        </div>
    </div>
</template>

<script>
import axios from 'axios'

export default {
    name: 'FileUpload',
    data() {
        return {
            file: '',
            dragging: false
        }
    },
    methods: {
        uploadFile(e) {
            this.dragging = false

            const files = e.target.files || e.dataTransfer.files
            const file = files?.[0]

            // Validate uploaded file:
            if(!file) {
                return
            } else if(!file.type.match('image')) {
                alert('please select an image file')
                return
            } else if(file.size > 5000000) { // Limit file size to 5MB
                alert('please check file size no over 5 MB.')
                return
            }

            this.file = file

            // N.b. we need to use `FormData` and set `Content-Type` in order for API to handle file:
            const formData = new FormData()
            formData.append('file', this.file)
            axios.post(
                'http://localhost:5000/file-upload',
                formData,
                {
                    headers: {
                        'Content-Type': 'multipart/form-data'
                    }
                })
                .then((response) => {
                    // TODO: emit successful response:
                    console.log('----- response: -----')
                    console.log(response)
                })
                .catch((error) => {
                    console.error(error)
                })
        },
        removeFile() {
            this.file = ''
        }
    }
}
</script>

<style scoped lang="scss">
    .file-upload {
        &__dropzone {
           width: 80%;
            height: 200px;
            position: relative;
            border-radius: 5px;
            border: 2px dashed rgba(0,0,0,.12);
            transition: 130ms ease-in-out border-color;

            &:hover {
                border-color: $color-primary;
            }
        }

        &__input {
            position: absolute;
            cursor: pointer;
            top: 0;
            right: 0;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
        }

        &__info {
            color: #A8A8A8;
            position: absolute;
            top: 50%;
            width: 100%;
            transform: translate(0, -50%);
            text-align: center;
        }

        &__icon {
            width: 100px;
            height: auto;
            opacity: 0.5;
        }

        &__title {
            color: #787878;
        }

        &__upload-limit-info {
            display: flex;
            justify-content: flex-start;
            flex-direction: column;
        }

        &__over {
            background: #5C5C5C;
            opacity: 0.8;
        }

        &__uploaded {
            width: 80%;
            height: 200px;
            position: relative;
            border: 2px dashed #eee;
        }

        &__uploaded-info {
            display: flex;
            flex-direction: column;
            align-items: center;
            color: #A8A8A8;
            position: absolute;
            top: 50%;
            width: 100%;
            transform: translate(0, -50%);
            text-align: center;
        }
    }
</style>
