<template>
    <v-card elevation="3" outlined :loading="isLoading">
        <!-- v-list-items and v-slider ensures consistent padding for card; otherwise, the loading bar would
        cause a jump in spacing when loading vs not loading: -->
        <v-list>
            <v-list-item class="py-2 px-6">
                <v-list-item-content>
                    <div
                        class="file-upload__dropzone"
                        :class="{'file-upload__dropzone--over': isDragging}"
                        @dragenter="isDragging = true"
                        @dragleave="isDragging = false"
                    >
                        <!-- ----- Initial state: ----- -->
                        <div v-if="!file">
                            <div class="file-upload__info" @drag="uploadFile">
                                <img class="file-upload__icon" src="/upload_file.svg" alt="upload icon">
                                <div class="pt-2">
                                    <span>Drag image here</span>
                                    <a>or browse</a>
                                </div>
                            </div>
                            <input class="file-upload__input" type="file" @change="uploadFile">
                        </div>
                        <!-- ----- Uploaded state: ----- -->
                        <div v-else>
                            <div class="file-upload__uploaded-info">
                                <span class="file-upload__title">Uploaded</span>
                                <v-btn color="primary" @click="removeFile">
                                    Remove File
                                </v-btn>
                            </div>
                        </div>
                    </div> <!-- /.file-upload__dropzone -->
                </v-list-item-content>
            </v-list-item>
        </v-list>
        <!-- No file selected: -->
    </v-card>
</template>

<script>
import axios from 'axios'

export default {
    name: 'FileUpload',
    data() {
        return {
            file: '',
            isDragging: false,
            isLoading: false
        }
    },
    methods: {
        uploadFile(e) {
            this.isDragging = false

            const files = e.target.files || e.dataTransfer.files
            const file = files?.[0]

            // Validate uploaded file:
            if(!file) {
                return
            } else if(!file.type.match('image')) {
                // TODO: replace alerts
                alert('please select an image file')
                return
            } else if(file.size > 5000000) { // Limit file size to 5MB
                // TODO: replace alerts
                alert('please check file size no over 5 MB.')
                return
            }

            this.file = file
            this.isLoading = true
            this.$emit('set-is-loading', true)

            // N.b. we need to use `FormData` and set `Content-Type` in order for API to handle file;
            // see <https://stackoverflow.com/q/43013858>:
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
                    this.$emit('get-products', {
                        'file-paths': response?.data?.file_paths
                    })
                })
                .catch((error) => {
                    console.error(error)
                })
                .finally(() => {
                    this.isLoading = false
                    this.$emit('set-is-loading', false)
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
            height: 110px;
            position: relative;
            border-radius: 5px;
            border: 3px dashed rgba(0, 0, 0, 0);
            transition: 150ms ease-in-out border-color;

            &:hover,
            &.file-upload__dropzone--over {
                border-color: $color-gray;
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
            height: auto;
            opacity: 0.3;
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
